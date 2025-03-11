#!/bin/bash
# Bash script to POST candidate data using curl

# Define JSON payload using a here-document
read -r -d '' JSON_DATA <<'EOF'
{
   "candidates":[
      {
         "id":1,
         "url":"https://cadencebank.com/",
         "actions":[
            {
               "selectOptions":"None"
            }
         ],
         "scan_domain":"www.cadencebank.com"
      },
      {
         "id":2,
         "url":"https://cadencebank.com/log-in",
         "actions":"None",
         "scan_domain":"www.cadencebank.com"
      },
      {
         "id":3,
         "url":"https://cadencebank.com/personal/digital-banking/online-banking",
         "actions":"None",
         "scan_domain":"www.cadencebank.com"
      }
   ]
}

EOF

# Execute the curl POST request
curl -X POST "http://127.0.0.1:4050/api/login_candidates" \
     -H "Content-Type: application/json" \
     -d "$JSON_DATA"



# "candidates": [
#     {
#       "id": 4,
#       "url": "https://www.hancockwhitney.com/insights/manage-risk-in-your-investment-portfolio",
#       "actions": null,
#       "scan_domain": "www.hancockwhitney.com"
#     },
#     {
#       "id": 5,
#       "url": "https://rewards.hancockwhitney.com/welcome.htm?login_error=true&product=0175",
#       "actions": [
#         {
#           "selectOptions": [
#             {"identifier": "account-select-17020485599316", "value": "CR"},
#             {"identifier": "account-select-169766242632384", "value": "CR"},
#             {"identifier": "dropdown-content", "value": "content1"}
#           ]
#         },
#         {
#           "step": 1,
#           "clickPosition": {"x": 1161, "y": 24},
#           "elementHTML": "<button class=\"hhs-menu-button hamburger-icon js-toggle-main-nav\" id=\"hamburger-icon\" style=\"min-width: 24px; min-height: 24px; display: block;\">...</button>",
#           "screenshot": "/app/modules/loginpagedetection/screenshot_flows/www_hancockwhitney_com/flow_4/page_1.png",
#           "url": "https://www.hancockwhitney.com/"
#         },
#         {
#           "step": 2,
#           "clickPosition": {"x": 1070, "y": 296},
#           "elementHTML": "<button onclick=\"handleSignOnClick(this);\" type=\"button\" class=\"button button--special-arrow ob-login__sign-in-button\">Go to Account Login</button>",
#           "screenshot": "/app/modules/loginpagedetection/screenshot_flows/www_hancockwhitney_com/flow_4/page_2.png",
#           "url": "https://www.hancockwhitney.com/"
#         },
#         {
#           "step": 3,
#           "clickPosition": {"x": 1218, "y": 70},
#           "elementHTML": "<a href=\"../externalLogin.htm\" class=\"btn btn-primary\" unselectable=\"on\">Log In</a>",
#           "screenshot": "/app/modules/loginpagedetection/screenshot_flows/www_hancockwhitney_com/flow_4/page_3.png",
#           "url": "https://rewards.hancockwhitney.com/welcome.htm?product=0175&ext_cat=0175A"
#         },
#         {
#           "step": 4,
#           "clickPosition": {"x": 639, "y": 415},
#           "elementHTML": "<button type=\"submit\" class=\"btn btn-primary btn-lg primary login m-btn\" onclick=\"this.disabled=1;$(this).addClass('btn-busy');this.innerHTML='<i class=\"fa fa-circle-notch fa-spin\"></i>';\">...</button>",
#           "screenshot": "/app/modules/loginpagedetection/screenshot_flows/www_hancockwhitney_com/flow_4/page_4.png",
#           "url": "https://rewards.hancockwhitney.com/externalLogin.htm"
#         }
#       ],
#       "scan_domain": "www.hancockwhitney.com"
#     }
#   ]