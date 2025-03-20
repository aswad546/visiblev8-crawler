const { URL } = require('url');
const puppeteer = require('puppeteer-extra');
const PuppeteerHar = require('puppeteer-har');
const { TimeoutError } = require('puppeteer-core');
const PuppeteerExtraPluginStealth = require('puppeteer-extra-plugin-stealth');
const pupp = require('puppeteer');
const fs = require( 'fs' );
// Tuning parameters
// Per Juestock et al. 30s + 15s for page load
const DEFAULT_NAV_TIME = 600;
const DEFAULT_LOITER_TIME = 15;

const sleep = ms => new Promise(r => setTimeout(r, ms));

const triggerClickEvent = async (page) => {
    await page.evaluate(() => {
        document.body.click();
    }); 
}

const triggerFocusBlurEvent = async (page) => {

    const inputElements = await page.$$('input');

    for (const input of inputElements) {
        try {
      // Scroll the element into view
      await page.evaluate((element) => {
        element.scrollIntoView({ behavior: 'smooth', block: 'center', inline: 'center' });
      }, input);

      // Wait for the element to be visible and clickable
      await page.waitForSelector('input', { visible: true, timeout: 1500});

      // Click the element
      await input.click({timeout:0});
      console.log('Clicked input element');

      // Optionally wait a bit between clicks
    //   await page.waitForTimeout(500);
    } catch (error) {
      console.log('Error clicking input element');
    }
  }
    //To trigger blur event
    // await page.click('body')
}

const triggerDoubleClickEvent = async(page) => {
    await page.evaluate(() => {
        const element = document.querySelector('body'); // Replace 'body' with any valid selector for the element you want to double-click.
        if (element) {
            const event = new MouseEvent('dblclick', {
                bubbles: true,
                cancelable: true,
                view: window
            });
            console.log('Attempting to trigger double click event handlers')
            element.dispatchEvent(event);
        }
    });
    
}
const triggerMouseEvents = async (page) => {
    // Simulate other mouse events on the first input
    try {
        const body = await page.$$('body');
    
        const box = await body[0].boundingBox();

        // Mouse move to the element
        await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, {timeout: 60000});

        // Mouse down event
        await page.mouse.down({timeout: 60000});

        // Mouse up event
        await page.mouse.up({timeout: 60000});

        // Mouse enter event doesn't directly exist, but moving the mouse to the element simulates it
        await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2, {timeout: 60000});
    } catch (e) {
        console.log('Error occured while trying to trigger mousemove event: ' + e);
    }
    
}


const triggerKeyEvents = async (page) => {
    await page.keyboard.press('Tab', { delay: 100 });
    await page.keyboard.down('Shift');
    await page.keyboard.press('!');  // Example to show Shift+1 -> !
    await page.keyboard.up('Shift');
}
const triggerCopyPasteEvents = async (page) => {
    await page.keyboard.down('Control');
    await page.keyboard.press('C'); // Assuming Windows/Linux. Use 'Meta' for macOS.
    await page.keyboard.up('Control');

    await page.keyboard.down('Control');
    await page.keyboard.press('V'); // Assuming Windows/Linux. Use 'Meta' for macOS.
    await page.keyboard.up('Control');
}

const triggerScrollEvent = async (page) => {
    const scrollStep = 100; // 100 px per step
    const scrollInterval = 100; // ms between each scroll
    let lastPosition = 0;
    let newPosition = 0;

    while (true) {
        newPosition = await page.evaluate((step) => {
            window.scrollBy(0, step);
            return window.pageYOffset;  // Get the new scroll position
        }, scrollStep);

        // If no more scrolling is possible, break the loop
        if (newPosition === lastPosition) {
            break;
        }

        lastPosition = newPosition;
        await sleep(scrollInterval);  // Wait before the next scroll
    }

    // Optionally scroll up or down using mouse if necessary
    try {
        await page.mouse.wheel({ deltaY: -100, timeout: 0 });  // Ensure enough timeout if mouse interaction is needed
    } catch (error) {
        console.error("Mouse wheel error:", error);
    }
}

const triggerWindowResize = async (page) => {
   
    const landscape = { width: 1280, height: 1000 };
    await page.setViewport(landscape);
    console.log('Set to landscape');

}
const triggerOrientationChangeEvents = async (page) => {
    // Dispatch an orientation change event
    await page.evaluate(() => {
      // Simulate changing to landscape
      Object.defineProperty(screen, 'orientation', {
          value: { angle: 90, type: 'landscape-primary' },
          writable: true
      });

      // Create and dispatch the event
      const event = new Event('orientationchange');
      window.dispatchEvent(event);
  });



}
const triggerTouchEvents = async (page) => {
        // Function to dispatch touch events
        async function dispatchTouchEvent(type, x, y) {
            await page.evaluate((type, x, y) => {
                function createTouch(x, y) {
                    return new Touch({
                        identifier: Date.now(),
                        target: document.elementFromPoint(x, y),
                        clientX: x,
                        clientY: y,
                        radiusX: 2.5,
                        radiusY: 2.5,
                        rotationAngle: 10,
                        force: 0.5,
                    });
                }
    
                const touchEvent = new TouchEvent(type, {
                    cancelable: true,
                    bubbles: true,
                    touches: [createTouch(x, y)],
                    targetTouches: [],
                    changedTouches: [createTouch(x, y)],
                    shiftKey: true,
                });
    
                const el = document.elementFromPoint(x, y);
                if (el) {
                    el.dispatchEvent(touchEvent);
                }
            }, type, x, y);
        }

        await dispatchTouchEvent('touchstart', 100, 150);

    // Simulate a touchmove
        await dispatchTouchEvent('touchmove', 105, 155);

        // Simulate a touchend
        await dispatchTouchEvent('touchend', 105, 155);

    
}

async function fillInputFields(page) {
    const inputElements = await page.$$('input');
  
    for (let input of inputElements) {
      try {
        await input.evaluate((element) => element.scrollIntoView());
        await sleep(Math.floor(Math.random() * 500) + 500); // Random delay after scrolling
  
        const isVisible = await input.evaluate((element) => {
          const style = window.getComputedStyle(element);
          return style && style.visibility !== 'hidden' && style.display !== 'none';
        });
  
        const isReadOnly = await input.evaluate((element) => element.hasAttribute('readonly'));
        const isDisabled = await input.evaluate((element) => element.hasAttribute('disabled'));
  
        if (isVisible && !isReadOnly && !isDisabled) {
          await Promise.race([
            input.type('aa', { delay: 100 }),
            new Promise((_, reject) => setTimeout(() => reject('Timeout'), 3000)),
          ]);
          // console.log('Successfully filled input field');
        } else {
          // console.log('Skipping non-interactable input field.');
        }
      } catch (e) {
        // console.log('Skipping input field due to timeout or other error:', e.message);
      }
    }
  
    await page.evaluate(() => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    await sleep(Math.floor(Math.random() * 500) + 500); // Random delay after scrolling to top
  
    await sleep(1000);
  }

const triggerEventHandlers = async (page) => {
    await sleep(1000) 
    console.log('Triggering the click event')
    await triggerClickEvent(page)
    console.log('Triggering double click event')
    await triggerDoubleClickEvent(page)
    console.log('Triggering the focus blur event')
    await triggerFocusBlurEvent(page)
    console.log('Triggering mouse events')
    await triggerMouseEvents(page)
    console.log('Triggering keyboard events')
    await triggerKeyEvents(page)
    console.log('Triggering copy/paste events')
    await triggerCopyPasteEvents(page)
    console.log('Triggering scroll/wheel events')
    await triggerScrollEvent(page)
    console.log('Triggering resize events')
    await triggerWindowResize(page)
    console.log('Triggering orientation events')
    await triggerOrientationChangeEvents(page)
    console.log('Triggering touch events')
    await triggerTouchEvents(page)
}

const configureConsentOMatic = async (browser) => {
    const page = await browser.newPage();
    
    // Navigate directly to the options page
    await page.goto('chrome-extension://pogpcelnjdlchjbjcakalbgppnhkondb/options.html');
    console.log('Gone to consent-o-matic configuration page')
    // Check if the correct page is loaded
    if (page.url() === 'chrome-extension://pogpcelnjdlchjbjcakalbgppnhkondb/options.html') {
        console.log('On the correct options page');

        // Set all sliders to true to accept all possible consent banners
        await page.evaluate(() => {
            let sliders = document.querySelectorAll('.categorylist li .knob');
            sliders.forEach(slider => {
                document.querySelector
                if (slider.ariaChecked === 'false') {
                    slider.click(); // This toggles the checkbox to 'true' if not already set
                }
            });
            //Go to dev tab of consent-o-matic extension settings
            let devTab = document.querySelector('.debug');
            devTab.click()
            //Enable the debug log if not already enabled 
            let debugClicksSlider = document.querySelector('[data-name=debugClicks]')
            if (debugClicksSlider.className !== 'active') {
                debugClicksSlider.click()
            }

            let skipHideMethodSlider = document.querySelector('[data-name=skipHideMethod]')
            if (skipHideMethodSlider.className !== 'active') {
                skipHideMethodSlider.click()
            }

            let dontHideProgressDialogSlider = document.querySelector('[data-name=dontHideProgressDialog]')
            if (dontHideProgressDialogSlider.className !== 'active') {
                dontHideProgressDialogSlider.click()
            }
            
            let debugLogSlider = document.querySelector('[data-name=debugLog]')
            if (debugLogSlider.className !== 'active') {
                debugLogSlider.click()
            }

        });


        console.log('All sliders set to true');
    } else {
        console.log('Not on the correct page, check the URL');
    }
    
    await page.close();
}

const registerConsentBannerDetector = async (page) => {
    return new Promise( (resolve, reject) => {
        const consoleHandler = msg => {

            if (msg.text() === '') { return }
    
            const text = msg.text()
            console.log(`- Console message: ${text}, [${msg.location().url}]`)
    
            if (msg.location().url.match(/moz-extension.*\/ConsentEngine.js/g)) {
                let matches = (/CMP Detected: (?<cmpName>.*)/).exec(text)
                if (matches) {
                    console.log(`- CMP found (${matches.groups.cmpName})`, worker)
                    resolve('Found')
                } else {
                    let matches = (/^(?<cmpName>.*) - (SAVE_CONSENT|HIDE_CMP|DO_CONSENT|OPEN_OPTIONS|Showing|isShowing\?)$/).exec(text)
                    if (matches) {
                        console.log('LOOKIE HEREEEEEEE matches:', matches)
                        // if (matches.contains('SAVE_CONSENT')) {
                        //     console.log
                        // }
                        console.log(`- CMP found (${matches.groups.cmpName})`, worker)
                        resolve('Found')
                    } else if (text.match(/No CMP detected in 5 seconds, stopping engine.*/g)) {
                        resolve('Not Found')
                    }
                }
            }
        }

        page.on('console', consoleHandler)

        setTimeout(() => {
            console.log('Removing event listeners')
            page.removeAllListeners('console')
            resolve('Not Found')
        }, 300000)
    })
    
}

/**
 * Given a selectOptions array (each element expected to have a "value" property),
 * find all select elements on the page in order and set their value to the corresponding
 * value from the array.
 * If there are fewer select elements than provided options, only the matching number are set.
 * Any errors encountered are logged using console.error.
 */
async function selectOptions(page, selectOptionsArray) {
    if (!selectOptionsArray) {
        console.log("No select options provided, skipping selection.");
        return;
      }
    
    try {
      const selects = await page.$$('select');
      if (!selects || selects.length === 0) {
        console.error('No select elements found on the page.');
        return;
      }
      // Set values for as many select elements as we have values
      const count = Math.min(selects.length, selectOptionsArray.length);
      for (let i = 0; i < count; i++) {
        try {
          await selects[i].evaluate((selectEl, newValue) => {
            try {
              selectEl.value = newValue;
              selectEl.dispatchEvent(new Event('change', { bubbles: true }));
            } catch (innerError) {
              console.error('Error setting value on select element:', innerError);
            }
          }, selectOptionsArray[i].value);
        } catch (err) {
          console.error(`Error processing select element index ${i}:`, err);
        }
      }
    } catch (e) {
      console.error('Error in selectOptions function:', e);
    }
  }

 /**
 * Executes the given list of actions on the page.
 * For each action that has a valid clickPosition (x, y),
 * it moves the mouse to that coordinate, clicks, and then waits
 * for any navigation or new tab. If a new page is detected, the function
 * switches to that page.
 * Errors are logged using console.error.
 *
 * @param {object} page - The Puppeteer page object.
 * @param {Array} actions - The list of action objects.
 * @returns {object} - The final page object (in case it changed due to navigation or new tab).
 */
async function executeActions(page, actions) {
    for (let action of actions) {
      if (!action.clickPosition || typeof action.clickPosition.x !== 'number' || typeof action.clickPosition.y !== 'number') {
        console.error("Skipping action due to missing or invalid clickPosition:", action);
        continue;
      }
  
      const { x, y } = action.clickPosition;
      try {
        // Move the mouse to the specified coordinates
        await page.mouse.move(x, y, { delay: 100 });
        await sleep(500);
        
        // Click at the specified coordinates
        await page.mouse.click(x, y);
        console.log(`Clicked at (${x}, ${y})`);
  
        // Wait for navigation or new tab after the click
        const newPage = await detectNavigationOrNewTab(page);
        if (newPage && newPage !== page) {
          console.log('Navigation or new tab detected after click. Switching to new page.');
          await page.close();
          page = newPage;
          await page.bringToFront();
        }
        // Wait a bit after the click for any effects to settle
        await sleep(1000);
      } catch (err) {
        console.error(`Error executing action at (${x}, ${y}):`, err);
      }
    }
    return page;
  }
  

/**
 * Detects if a navigation or new tab has been opened after a click.
 * Returns the new page if detected, or null if no navigation/new tab occurred within the timeout.
 */
async function detectNavigationOrNewTab(page) {
    const timeout = 10000;
    const browser = page.browser();
  
    return Promise.race([
      page.waitForNavigation({ timeout }).then(() => {
        console.log('Navigation detected.');
        return page;
      }).catch(() => null),
      new Promise((resolve) => {
        const listener = async (target) => {
          if (target.opener() === page.target()) {
            const newPage = await target.page();
            await newPage.bringToFront();
            console.log('New tab detected.');
            browser.off('targetcreated', listener);
            resolve(newPage);
          }
        };
        browser.on('targetcreated', listener);
        setTimeout(() => {
          browser.off('targetcreated', listener);
          resolve(null);
        }, timeout);
      })
    ]);
  }
  


// CLI entry point
function main() {
    const { program } = require('commander');
    const default_crawler_args = [
                    "--disable-setuid-sandbox",
                    "--no-sandbox",
                    '--enable-logging=stderr',
                    '--enable-automation',
                    //'--v=1'
                    // '--disable-extensions-except=/app/node/consent-o-matic',
                    // '--load-extension=/app/node/consent-o-matic',
                ];
    // The raw extra arguments (after the first 5) might include additional crawler arguments
    let rawArgs = process.argv.slice(5);


    
    program
        .version('1.0.0');
    program
        .command("visit <URL> <uid>")
        .option( '--headless <headless>', 'Which headless mode to run visiblev8', 'new')
        .option( '--loiter-time <loiter_time>', 'Amount of time to loiter on a webpage', DEFAULT_LOITER_TIME)
        .option( '--nav-time <nav_time>', 'Amount of time to wait for a page to load', DEFAULT_NAV_TIME)
        .option('--actions <actions>', 'JSON formatted actions', "")  // Added actions option
        .allowUnknownOption(true)
        .description("Visit the given URL and store it under the UID, creating a page record and collecting all data")
        .action(async function(input_url, uid, options) {
            // Remove any "--actions" flag and its value from the raw arguments.
            // This ensures that the extra arguments contain only crawler options.
            let filteredArgs = [];
            for (let i = 0; i < rawArgs.length; i++) {
                if (rawArgs[i] === '--actions') {
                i++; // Skip the next argument (the JSON string)
                } else {
                filteredArgs.push(rawArgs[i]);
                }
            }


            let combined_crawler_args = default_crawler_args.concat(filteredArgs);
            let show_log = false;
            const user_data_dir = `/tmp/${uid}`;

            if ( combined_crawler_args.includes( '--show-chrome-log' ) ) {
                show_log = true;
                combined_crawler_args = combined_crawler_args.filter( function( item ) {
                    return item !== '--show-chrome-log';
                } );
            }

            if ( combined_crawler_args.includes( '--no-headless' ) ) {
                options.headless = false;
                combined_crawler_args = combined_crawler_args.filter( function( item ) {
                    return item !== '--no-headless';
                } );
            }

            if ( combined_crawler_args.includes( '--no-screenshot' ) ) {
                options.disable_screenshots = true;
                combined_crawler_args = combined_crawler_args.filter( function( item ) {
                    return item !== '--no-screenshot';
                } );
            }

            if ( show_log ) {
                console.log( `Using user data dir: ${user_data_dir}` );
                console.log(`Running chrome with, screenshot set to ${!options.disable_screenshots}, 
                headless set to ${options.headless} 
                and args: ${combined_crawler_args.join(' ')} to crawl ${input_url}`)
            }

                // Parse the actions JSON if provided via the --actions option.
            let actions = null;
            console.log(`actions: ${options.actions}`)
            if (options.actions) {
                try {
                    actions = JSON.parse(options.actions);
                    console.log("Actions received:", actions);
                } catch (err) {
                    console.error("Invalid JSON provided for --actions" + err);
                    process.exit(1);
                }
            }

            puppeteer.use(PuppeteerExtraPluginStealth());
            const browser = await puppeteer.launch({
                headless: true,
                userDataDir: user_data_dir,
                dumpio: show_log,
                executablePath: '/opt/chromium.org/chromium/chrome',
                args: combined_crawler_args,
                timeout: 600 * 1000,
                protocolTimeout: 600 * 1000,
            });

            // await configureConsentOMatic(browser)
 
            console.log('Launching new browser')

            let page = await browser.newPage( { viewport: null } );
            await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64)');
            await page.setViewport({ width: 1280, height: 800 });
            await page.evaluateOnNewDocument(() => {
                delete navigator.__proto__.webdriver;
            });
            console.log('Created new page')
            const har = new PuppeteerHar(page);
            const url = new URL(input_url);
            console.log('Visiting url: ' + url)
            console.log(options)
            try {
                await har.start({ path: `${uid}.har` });
                try{
                    const navigationPromise = page.goto(url, {
                        timeout: options.navTime * 1000,
                        waitUntil: 'load'
                    });

                    // const consentBannerPromise = registerConsentBannerDetector(page)
                    //Wait for the page to load and the consent banner to be triggered
                    await Promise.all([ navigationPromise])
                    console.log('Page load event is triggered')
                    //Wait for any additional scripts to load
                    await sleep(4000)


                    /**
                     * {
                        "id":1,
                        "url":"https://www.hancockwhitney.com/",
                        "actions":[
                            {
                                "selectOptions":[
                                {
                                    "identifier":"account-select-17020485599316",
                                    "value":"Select an Account Type"
                                },
                                {
                                    "identifier":"account-select-169766242632384",
                                    "value":"Select an Account Type"
                                },
                                {
                                    "identifier":"dropdown-content",
                                    "value":"content2"
                                }
                                ]
                            }
                        ],
                        "scan_domain":"www.hancockwhitney.com"
                    },
                     */
                    // Only execute actions if there is atleast one action in the input actions
                    // Example input can be found above for actions
                    if (actions && actions.length > 0) {
                        // Extract the first action object that should contain the selectOptions.
                        const firstAction = actions.shift();
                        // If selectOptions is not null, then perform the selection.
                        if (firstAction.selectOptions !== null) {
                          console.log('Selecting Options');
                          await selectOptions(page, firstAction.selectOptions);
                        } else {
                          console.log('No select options provided in actions; skipping selection.');
                        }
                        // Continue with the remaining actions.
                        if (actions.length > 0) {
                          console.log('Executing Actions');
                          page = await executeActions(page, actions);
                        }
                      }
                      
                    await triggerEventHandlers(page)
                    console.log('Triggered all events: ' + input_url)
                    await sleep(options.loiterTime * 1000);
                } catch (ex) {
                    if ( ex instanceof TimeoutError ) {
                        console.log('TIMEOUT OCCURED WHILE VISITING: ' + url)
                        // await sleep(options.loiterTime * 1000 * 2);
                        throw ex;
                    } else {
                        console.log(`Error occured ${ex} while visiting: ` + url)
                        throw ex;
                    }
                }
                if ( !options.disable_screenshots )
                    await page.screenshot({path: `./${uid}.png`, fullPage: true, timeout: 0 });

                    
            } catch (ex) {
                if (ex.message != 'Found or Not Found') {
                    console.log(`FAILED CRAWL TO ${url}`)
                    console.error(ex);
                    process.exitCode = -1;
                }
                
            }
            console.log( 'Pid of browser process', browser.process().pid )
            await har.stop()
            await page.close();
            await browser.close();
            console.log(`Finished crawling, ${url} cleaning up...`);
            // Throw away user data
            await fs.promises.rm(user_data_dir, { recursive: true, force: true });
            
        });
    program.parse(process.argv);
}

main();
