# ResNet-50 Express Server

Express server that accepts image uploads via POST requests and returns classification results (label and probability) using the ResNet-50 model.

## Setup

1. Install Node.js dependencies:
```bash
npm install
```

2. Make sure you have Python and the required Python packages installed:
```bash
pip install transformers torch pillow
```

## Running the Server

Start the server:
```bash
npm start
```

Or for development with auto-reload:
```bash
npm run dev
```

The server will start on `http://localhost:3000` (or the port specified in the PORT environment variable).

## API Endpoints

### POST /classify

Classify an uploaded image.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: Form data with an `image` field containing the image file

**Response:**
```json
{
  "label": "golden retriever",
  "probability": 0.987654321,
  "classId": 207
}
```

**Example using curl (Linux/Mac/Git Bash):**
```bash
curl -X POST http://localhost:3000/classify \
  -F "image=@your_image.jpg"
```

fetch('http://localhost:3000/classify', {
  method: 'POST',
  body: formData
})
.then(response => response.json())
.then(data => {
  console.log('Label:', data.label);
  console.log('Probability:', data.probability);
});
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## Error Handling

The server returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (no image provided or invalid file type)
- `500`: Server error (classification failed)

## Notes

- Maximum file size: 10MB
- Supported image formats: Any format supported by PIL (JPEG, PNG, etc.)
- Temporary image files are automatically cleaned up after processing

