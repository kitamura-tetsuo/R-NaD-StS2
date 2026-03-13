write comments in english.

## Screenshot API
You can take a screenshot of the game by calling the following API:
`GET http://127.0.0.1:8081/screenshot`

The screenshot will be saved to `./tmp/screenshot_<timestamp>.png` and the path will be returned in the JSON response.

### Headless Support
This API supports taking screenshots even in headless mode (using Godot's internal viewport capture). If the internal capture fails, it will fall back to a standard screen grab.