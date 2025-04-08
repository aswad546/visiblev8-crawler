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

// Logging helper to ensure URL is included in all logs
let currentUrl = ""; // Will be set when crawl starts

const logger = {
    log: (message, url = currentUrl) => {
        console.log(`[URL:${url}] ${message}`);
    },
    error: (message, error = null, url = currentUrl) => {
        const errorMsg = error ? `${message}: ${error}` : message;
        console.error(`[URL:${url}] ERROR: ${errorMsg}`);
    },
    warn: (message, url = currentUrl) => {
        console.warn(`[URL:${url}] WARN: ${message}`);
    }
};

const sleep = ms => new Promise(r => setTimeout(r, ms));

const triggerClickEvent = async (page) => {
    try {
        await page.evaluate(() => {
            document.body.click();
        }); 
        
        // Check for navigation after click
        const newPage = await detectNavigationOrNewTab(page);
        if (newPage !== null) {
            logger.log('Navigation detected after click event');
            return newPage;
        }
        return page;
    } catch (e) {
        logger.error('Error in triggerClickEvent', e);
        return page;
    }
}

const triggerFocusBlurEvent = async (page) => {
    try {
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
                logger.log('Clicked input element');

                // Check for navigation after input click
                const newPage = await detectNavigationOrNewTab(page);
                if (newPage !== null && newPage !== page) {
                    logger.log('Navigation detected after input click');
                    return newPage;
                }
            } catch (error) {
                logger.log('Error clicking input element');
            }
        }
        return page;
    } catch (e) {
        logger.error('Error in triggerFocusBlurEvent', e);
        return page;
    }
}

const triggerDoubleClickEvent = async(page) => {
    try {
        await page.evaluate(() => {
            const element = document.querySelector('body');
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
        
        // Check for navigation after double click
        const newPage = await detectNavigationOrNewTab(page);
        if (newPage !== null) {
            logger.log('Navigation detected after double click event');
            return newPage;
        }
        return page;
    } catch (e) {
        logger.error('Error in triggerDoubleClickEvent', e);
        return page;
    }
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
        logger.error('Error occurred while trying to trigger mousemove event', e);
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
        logger.error("Mouse wheel error", error);
    }
}

const triggerWindowResize = async (page) => {
   
    const landscape = { width: 1280, height: 1000 };
    await page.setViewport(landscape);
    logger.log('Set to landscape');

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
          // logger.log('Successfully filled input field');
        } else {
          // logger.log('Skipping non-interactable input field.');
        }
      } catch (e) {
        // logger.log('Skipping input field due to timeout or other error:', e.message);
      }
    }
  
    await page.evaluate(() => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    });
    await sleep(Math.floor(Math.random() * 500) + 500); // Random delay after scrolling to top
  
    await sleep(1000);
  }

  const triggerEventHandlers = async (page) => {
    await sleep(5000);

    try {
        logger.log('Triggering the click event');
        const pageAfterClick = await triggerClickEvent(page);
        if (pageAfterClick !== page) {
            page = pageAfterClick;
        }

    } catch (e) {
        logger.error('Error triggering click event', e);
    }

    try {
        logger.log('Triggering double click event');
        const pageAfterDoubleClick = await triggerDoubleClickEvent(page);
        if (pageAfterDoubleClick !== page) {
            page = pageAfterDoubleClick;
        }

    } catch (e) {
        logger.error('Error triggering double click event', e);
    }

    try {
        logger.log('Triggering the focus blur event');
        const pageAfterFocusBlur = await triggerFocusBlurEvent(page);
        if (pageAfterFocusBlur !== page) {
            page = pageAfterFocusBlur;
        }

    } catch (e) {
        logger.error('Error triggering focus blur event', e);
    }

    try {
        logger.log('Triggering mouse events');
        await triggerMouseEvents(page);
    } catch (e) {
        logger.error('Error triggering mouse events', e);
    }

    try {
        logger.log('Triggering keyboard events');
        await triggerKeyEvents(page);
    } catch (e) {
        logger.error('Error triggering keyboard events', e);
    }

    try {
        logger.log('Triggering copy/paste events');
        await triggerCopyPasteEvents(page);
    } catch (e) {
        logger.error('Error triggering copy/paste events', e);
    }

    try {
        logger.log('Triggering scroll/wheel events');
        await triggerScrollEvent(page);
    } catch (e) {
        logger.error('Error triggering scroll/wheel events', e);
    }

    try {
        logger.log('Triggering resize events');
        await triggerWindowResize(page);
    } catch (e) {
        logger.error('Error triggering resize events', e);
    }

    try {
        logger.log('Triggering orientation events');
        await triggerOrientationChangeEvents(page);
    } catch (e) {
        logger.error('Error triggering orientation events', e);
    }

    try {
        logger.log('Triggering touch events');
        await triggerTouchEvents(page);
    } catch (e) {
        logger.error('Error triggering touch events', e);
    }
    return page; // Return the final page reference
}

const configureConsentOMatic = async (browser) => {
    const page = await browser.newPage();
    
    // Navigate directly to the options page
    await page.goto('chrome-extension://pogpcelnjdlchjbjcakalbgppnhkondb/options.html');
    logger.log('Gone to consent-o-matic configuration page');
    // Check if the correct page is loaded
    if (page.url() === 'chrome-extension://pogpcelnjdlchjbjcakalbgppnhkondb/options.html') {
        logger.log('On the correct options page');

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


        logger.log('All sliders set to true');
    } else {
        logger.log('Not on the correct page, check the URL');
    }
    
    await page.close();
}

const registerConsentBannerDetector = async (page) => {
    return new Promise( (resolve, reject) => {
        const consoleHandler = msg => {

            if (msg.text() === '') { return }
    
            const text = msg.text()
            logger.log(`- Console message: ${text}, [${msg.location().url}]`);
    
            if (msg.location().url.match(/moz-extension.*\/ConsentEngine.js/g)) {
                let matches = (/CMP Detected: (?<cmpName>.*)/).exec(text)
                if (matches) {
                    logger.log(`- CMP found (${matches.groups.cmpName})`, worker);
                    resolve('Found');
                } else {
                    let matches = (/^(?<cmpName>.*) - (SAVE_CONSENT|HIDE_CMP|DO_CONSENT|OPEN_OPTIONS|Showing|isShowing\?)$/).exec(text)
                    if (matches) {
                        logger.log('LOOKIE HEREEEEEEE matches:', matches);
                        // if (matches.contains('SAVE_CONSENT')) {
                        //     console.log
                        // }
                        logger.log(`- CMP found (${matches.groups.cmpName})`, worker);
                        resolve('Found');
                    } else if (text.match(/No CMP detected in 5 seconds, stopping engine.*/g)) {
                        resolve('Not Found');
                    }
                }
            }
        }

        page.on('console', consoleHandler)

        setTimeout(() => {
            logger.log('Removing event listeners');
            page.removeAllListeners('console');
            resolve('Not Found');
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
        logger.log("No select options provided, skipping selection.");
        return;
      }
    
    try {
      const selects = await page.$$('select');
      if (!selects || selects.length === 0) {
        logger.error('No select elements found on the page.');
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
          logger.error(`Error processing select element index ${i}:`, err);
        }
      }
    } catch (e) {
      logger.error('Error in selectOptions function:', e);
    }
  }

  async function detectNavigationOrNewTab(page) {
    let timeoutId = null;
    let targetListener = null;
    let navigationPromise = null;
    
    try {
      const timeout = 5000;
      const browser = page.browser();
      
      // Create a cleanup function to remove all listeners
      const cleanup = () => {
        if (timeoutId) {
          clearTimeout(timeoutId);
          timeoutId = null;
        }
        if (targetListener && browser) {
          browser.off('targetcreated', targetListener);
          targetListener = null;
        }
      };
      
      return await new Promise((mainResolve) => {
        // First promise: navigation on the same page
        navigationPromise = page.waitForNavigation({ timeout })
          .then((result) => {
            logger.log('Navigation detected');
            cleanup();
            mainResolve(page);
            return page;
          })
          .catch(() => null);  // Just return null on timeout
        
        // Second promise: new tab detection
        targetListener = async (target) => {
          if (target.opener() === page.target()) {
            const newPage = await target.page();
            await newPage.bringToFront();
            logger.log('New tab detected');
            cleanup();
            mainResolve(newPage);
          }
        };
        
        browser.on('targetcreated', targetListener);
        
        // Set the timeout to handle the case where neither navigation nor new tab occurs
        timeoutId = setTimeout(() => {
          logger.log('Navigation/tab timeout reached');
          cleanup();
          mainResolve(null);
        }, timeout);
      });
    } catch (error) {
      logger.error('Error in detectNavigationOrNewTab:', error);
      return null;
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
        logger.error("Skipping action due to missing or invalid clickPosition:", action);
        continue;
      }
  
      const { x, y } = action.clickPosition;
      try {
        // Move the mouse to the specified coordinates
        await page.mouse.move(x, y, { delay: 100 });
        await sleep(500);
        
        // Click at the specified coordinates
        await page.mouse.click(x, y);
        logger.log(`Clicked at (${x}, ${y})`);
  
        // Wait for navigation or new tab after the click
        const newPage = await detectNavigationOrNewTab(page);
        
        // Handle the navigation result
        if (newPage === null) {
          logger.log('No navigation or new tab detected after click. Continuing with current page.');
        } else if (newPage !== page) {
          logger.log('New tab detected after click. Switching to new tab.');
          try {
            // Only close the original page if we successfully got a new page
            await page.close().catch(err => logger.error('Error closing original page:', err));
            page = newPage;
          } catch (err) {
            logger.error('Error switching to new tab:', err);
          }
        } else {
          logger.log('Navigation detected on current page.');
        }
        
        // Wait a bit after the action for any effects to settle
        await sleep(4000); // Look here mf
      } catch (err) {
        logger.error(`Error executing action at (${x}, ${y}):`, err);
      }
    }
    
    return page;
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
            // Set current URL to be used in all logging
            currentUrl = input_url;
            
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
            const user_data_dir = `/tmp/${uid}_${Date.now()}_${Math.random().toString(36).substring(2, 15)}`;;

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
                logger.log(`Using user data dir: ${user_data_dir}`);
                logger.log(`Running chrome with, screenshot set to ${!options.disable_screenshots}, 
                headless set to ${options.headless} 
                and args: ${combined_crawler_args.join(' ')} to crawl ${input_url}`);
            }

                // Parse the actions JSON if provided via the --actions option.
            let actions = null;
            logger.log(`actions: ${options.actions}`);
            if (options.actions) {
                try {
                    actions = JSON.parse(options.actions);
                    logger.log("Actions received: " + JSON.stringify(actions));
                } catch (err) {
                    logger.error("Invalid JSON provided for --actions", err);
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
 
            logger.log('Launching new browser');

            let page = await browser.newPage( { viewport: null } );
            await page.setUserAgent('Mozilla/5.0 (Windows NT 10.0; Win64; x64)');
            await page.setViewport({ width: 1280, height: 800 });
            await page.evaluateOnNewDocument(() => {
                delete navigator.__proto__.webdriver;
            });
            logger.log('Created new page');
            const har = new PuppeteerHar(page);
            const url = new URL(input_url);
            logger.log('Visiting url: ' + url);
            logger.log('Crawl options: ' + JSON.stringify(options));
            try {
                await har.start({ path: `${uid}.har` });
                try{
                    const navigationPromise = page.goto(url, {
                        timeout: options.navTime * 1000,
                        waitUntil: 'load'
                    });

                    // const consentBannerPromise = registerConsentBannerDetector(page)
                    //Wait for the page to load and the consent banner to be triggered
                    await Promise.all([ navigationPromise]);
                    logger.log('Page load event is triggered');
                    //Wait for any additional scripts to load
                    await sleep(4000);


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
                          logger.log('Selecting Options');
                          await selectOptions(page, firstAction.selectOptions);
                        } else {
                          logger.log('No select options provided in actions; skipping selection.');
                        }
                        // Continue with the remaining actions.
                        if (actions.length > 0) {
                          logger.log('Executing Actions');
                          await fillInputFields(page);
                          page = await executeActions(page, actions);
                          await page.screenshot({path: `./${uid}_actions.png`, fullPage: true, timeout: 0 });
                        }
                      }
                      
                    page = await triggerEventHandlers(page);
                    logger.log('Triggered all events');
                    await sleep(options.loiterTime * 1000);
                } catch (ex) {
                    if ( ex instanceof TimeoutError ) {
                        logger.error('TIMEOUT OCCURRED WHILE VISITING', ex);
                        // await sleep(options.loiterTime * 1000 * 2);
                        throw ex;
                    } else {
                        logger.error(`Error occurred while visiting`, ex);
                        throw ex;
                    }
                }
                if ( !options.disable_screenshots )
                    await page.screenshot({path: `./${uid}.png`, fullPage: true, timeout: 0 });

                    
            } catch (ex) {
                if (ex.message != 'Found or Not Found') {
                    logger.error(`FAILED CRAWL`, ex);
                    process.exitCode = -1;
                }
                
            }
            logger.log('Pid of browser process: ' + browser.process().pid);
            await har.stop();
            await page.close();
            await browser.close();
            logger.log(`Finished crawling, cleaning up...`);
            // Throw away user data
            await fs.promises.rm(user_data_dir, { recursive: true, force: true });
            
        });
    program.parse(process.argv);
}

main();