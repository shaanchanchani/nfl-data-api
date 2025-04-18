# NFL Data API

A FastAPI-based API for accessing and analyzing NFL data.

## Features

- Player statistics and analysis
- Team statistics
- Game analysis
- Historical matchup data
- Situation-based statistics
- And more...

## Local Development

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the development server:
```bash
uvicorn src.nfl_data.main:app --reload
```

## Deployment on Railway

### Option 1: Using Railway CLI (Recommended)

1. Install Railway CLI:
```bash
# macOS
brew install railway

# Windows (requires npm)
npm i -g @railway/cli

# Other platforms
curl -fsSL https://railway.app/install.sh | sh
```

2. Login to Railway:
```bash
railway login
```

3. Link your project:
```bash
# If you're creating a new project
railway init

# If you're connecting to an existing project
railway link
```

4. Deploy your application:
```bash
railway up
```

5. Monitor your deployment:
```bash
railway logs
```

### Option 2: Using Railway Dashboard

1. Create a new project on [Railway](https://railway.app)
2. Connect your GitHub repository
3. Railway will automatically detect the Python project and build it
4. The application will be deployed automatically

## Environment Variables

The following environment variables can be configured:

- `PORT`: The port number for the server (set automatically by Railway)
- Add any additional environment variables needed for your specific setup

To set environment variables using Railway CLI:
```bash
railway vars set KEY=VALUE
```

To view current environment variables:
```bash
railway vars
```

## Railway CLI Common Commands

```bash
# Start a development environment
railway run

# View project status
railway status

# Open project dashboard
railway open

# View deployment logs
railway logs

# List all services
railway service list

# Generate production build
railway build

# Deploy to production
railway up
```

## API Documentation

Once deployed, you can access the API documentation at:

- Swagger UI: `/docs`
- ReDoc: `/redoc`

## License

MIT