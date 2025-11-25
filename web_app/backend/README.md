# Backend

Flask API for image colorization predictions.

## Setup

```bash
pip install -r requirements.txt
python app.py
```

## Endpoints

- `GET /models` - Return a list of all available models.
- `POST /colourize` - Accepts a grayscale image and returns the colorized image, using the specified models.
