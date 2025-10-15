# Demo API Reference

## Authentication
Authenticate using API keys passed via the `Authorization` header.

## Endpoints
- `GET /status` returns service health.
- `POST /jobs` accepts a JSON body describing the job to run.
- `GET /jobs/{job_id}` retrieves job status and results.

## Rate Limits
Clients are limited to 120 requests per minute. Use exponential backoff when you receive a 429 response.
