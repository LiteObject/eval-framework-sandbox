# Python Requests Library Overview

The Python `requests` library is a popular HTTP library that makes it simple to send HTTP/1.1 requests. It provides methods for sending HTTP requests, handling responses, managing sessions, and dealing with headers and cookies.

## Installing Requests
To install the library, use `pip install requests` in your Python environment.

## Making a GET Request
You can make a GET request using `requests.get(url)`. The response object contains metadata and the content of the request.

```python
import requests
response = requests.get("https://api.example.com/data")
print(response.status_code)
print(response.json())
```

## Posting Data
Use `requests.post(url, data=data)` to send form data, or `json=payload` for JSON bodies.

## Sessions
Sessions allow you to persist parameters across requests. Instantiate a session with `session = requests.Session()` and use it to send requests.

## Error Handling
Requests raises exceptions like `requests.exceptions.Timeout` and `requests.exceptions.HTTPError`. Check `response.raise_for_status()` to raise for 4xx/5xx responses.
