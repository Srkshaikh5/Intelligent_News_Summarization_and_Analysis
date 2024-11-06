
# FastAPI News and Trending API

This FastAPI application provides endpoints to retrieve news articles based on a query and fetch trending topics. It uses Redis for caching to improve performance and reduce response times.

## Features
- **News Endpoint**: Retrieve news articles based on a query (e.g., `technology`).
- **Trending Topics Endpoint**: Get a list of trending topics.
- **Caching**: Utilizes Redis to cache responses for faster access.

## Requirements
- **Python** 3.8+
- **Redis** server (for caching)
- **FastAPI** for creating API endpoints
- **Uvicorn** as the ASGI server to run the FastAPI app
- **AWS CLI** For easy configuration of AWS credentials.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Srkshaikh5/Intelligent_News_Summarization_and_Analysis.git
cd Intelligent_News_Summarization_and_Analysis
```

### 2. Set Up Virtual Environment
It’s recommended to use a virtual environment to manage dependencies:
```bash
# Create virtual environment
python -m venv env

# Activate virtual environment
# On Windows
.\env\Scripts\activate
# On macOS/Linux
source env/bin/activate
```

### 3. Install Requirements
Install all necessary packages using `requirements.txt`:
```bash
pip install -r requirements.txt
```
### 4. Setting Keys and Secrets
  **ADD New API Key in .env file**
    - ```API_KEY= <youAPIKey>```
  **AWS Configuration**:
   - Ensure your AWS keys are set in the environment:
     ```bash
     export AWS_ACCESS_KEY_ID='<YourAWSAccessKey>'
     export AWS_SECRET_ACCESS_KEY='<YourAWSSecretKey>'
     export AWS_DEFAULT_REGION='<YourAWSRegion>'
     ```
   - Or configure them using the AWS CLI:
     ```bash
     aws configure
     ```

   **Environment Variables for Claude Model**:
   - The Claude model uses AWS credentials set in the environment variables above. Ensure these are active in your terminal before running the application.



### 5. Install and Run Redis
Redis is required to enable caching for the API. 

#### For Windows:
1. Download Redis from [https://github.com/microsoftarchive/redis/releases](https://github.com/microsoftarchive/redis/releases).
2. Extract and install Redis.
3. Run Redis by executing `redis-server` in the extracted folder.

#### For macOS (using Homebrew):
```bash
brew install redis
brew services start redis
```

#### For Linux:
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis
```

Verify Redis is running by using:
```bash
redis-cli ping
# Expected output: PONG
```

### 6. Run the FastAPI Application
Use `uvicorn` to start the FastAPI server:
```bash
uvicorn fastapi_app:app --reload
```
The app should now be running at [http://127.0.0.1:8000](http://127.0.0.1:8000).

### 7. API Endpoints

- **`GET /news/`**: Fetch news articles based on a `query` parameter.
  - Example: [http://127.0.0.1:8000/news/?query=technology](http://127.0.0.1:8000/news/?query=technology)
- **`GET /trending/`**: Fetch trending topics.
  - Example: [http://127.0.0.1:8000/trending/](http://127.0.0.1:8000/trending/)

### 8. Testing and Troubleshooting
If Redis is unavailable, the app will log a warning and skip caching. Ensure Redis is running if caching is needed.

## Project Structure
- `fastapi_app.py`: Main application file containing API endpoints and caching configuration.
- `requirements.txt`: Lists required Python packages.

## Notes
- The `@cache` decorator is applied to endpoints to cache responses for 60 seconds.
- Logging is enabled to monitor errors and warnings for debugging purposes.
