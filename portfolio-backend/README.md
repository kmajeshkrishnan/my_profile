# Portfolio Backend API

This is the backend API for the portfolio website that provides endpoints for image processing, text processing, and contact form functionality.

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements_base.txt
pip install -r requirements_heavy.txt
```

3. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Update the email credentials in `.env`

## Running the Server

```bash
uvicorn main:app --reload
```

The server will start at `http://localhost:8000`

## API Endpoints

### 1. Process Image
- **Endpoint**: `/process-image`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Description**: Processes an uploaded image and returns the modified version
- **Example**: 
```bash
curl -X POST -F "file=@image.jpg" http://localhost:8000/process-image
```

### 2. Process Text
- **Endpoint**: `/process-text`
- **Method**: POST
- **Content-Type**: application/json
- **Description**: Processes text using an LLM and returns the response
- **Example**:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"text":"Hello, world!"}' http://localhost:8000/process-text
```

### 3. Send Contact Email
- **Endpoint**: `/send-contact-email`
- **Method**: POST
- **Content-Type**: application/json
- **Description**: Sends a confirmation email to the contact form submitter
- **Example**:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"name":"John Doe","email":"john@example.com","message":"Hello!"}' http://localhost:8000/send-contact-email
```

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Security Notes

1. In production, update the CORS settings in `main.py` to only allow specific origins
2. Use environment variables for sensitive information
3. Consider implementing rate limiting for the endpoints
4. Use HTTPS in production 