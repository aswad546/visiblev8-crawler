#!/bin/bash

# Define the URL and the data
URL="http://172.17.0.1:8100/analyze"

# You can insert your IDs here (e.g., 2274, 2275, 2276, etc.)
# DATA="[2274,2275,2276,2277,2278,2279,2280,2281,2282,2283,2284,2285,2286,2287,2288,2289,2290,2291,2292,2293,2294,2295,2296,2297,2298,2299,2300,2301,2302,2303,2304,2305,2306,2307,2308,2309,2310,2311,2312,2313,2314,2315,2316,2317,2318,2319,2320]"
DATA="[2732]"
# Send the POST request with curl
RESPONSE=$(curl -s -w "%{http_code}" -o response.txt -X POST "$URL" -H "Content-Type: application/json" -d "$DATA")

# Check the HTTP status code
HTTP_STATUS=$(echo "$RESPONSE" | tail -n1)

if [ "$HTTP_STATUS" -eq 200 ]; then
    echo "POST request successful. Response saved in response.txt."
else
    echo "POST request failed. HTTP Status: $HTTP_STATUS"
    cat response.txt
fi
