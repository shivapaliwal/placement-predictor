# Deployment Guide

This guide covers deploying the Student Placement Predictor to various platforms.

## Prerequisites

1. **Model Training**: Ensure the model is trained and saved
   ```bash
   py model_training.py
   ```

2. **Dependencies**: Install production dependencies
   ```bash
   py -m pip install -r requirements.txt
   ```

## Local Development

```bash
# Development mode
py app.py

# Production mode
py wsgi.py
```

## Docker Deployment

### Build and Run with Docker

```bash
# Build the Docker image
docker build -t placement-predictor .

# Run the container
docker run -p 5000:5000 placement-predictor

# Run in detached mode
docker run -d -p 5000:5000 --name placement-app placement-predictor
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Run with:
```bash
docker-compose up -d
```

## Heroku Deployment

1. **Install Heroku CLI** and login:
   ```bash
   heroku login
   ```

2. **Create Heroku app**:
   ```bash
   heroku create your-app-name
   ```

3. **Set environment variables**:
   ```bash
   heroku config:set FLASK_ENV=production
   ```

4. **Deploy**:
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

5. **Open the app**:
   ```bash
   heroku open
   ```

## AWS Deployment

### EC2 with Docker

1. **Launch EC2 instance** (Ubuntu recommended)
2. **Install Docker**:
   ```bash
   sudo apt update
   sudo apt install docker.io
   sudo systemctl start docker
   sudo systemctl enable docker
   ```

3. **Deploy application**:
   ```bash
   sudo docker run -d -p 80:5000 --name placement-app placement-predictor
   ```

### AWS ECS

1. **Create ECR repository**:
   ```bash
   aws ecr create-repository --repository-name placement-predictor
   ```

2. **Build and push image**:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com
   docker tag placement-predictor:latest your-account.dkr.ecr.us-east-1.amazonaws.com/placement-predictor:latest
   docker push your-account.dkr.ecr.us-east-1.amazonaws.com/placement-predictor:latest
   ```

3. **Create ECS cluster and service** (via AWS Console or CLI)

## Google Cloud Platform

### Cloud Run

1. **Enable Cloud Run API**
2. **Deploy**:
   ```bash
   gcloud run deploy placement-predictor \
     --image gcr.io/PROJECT_ID/placement-predictor \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

## Azure Deployment

### Azure Container Instances

```bash
az container create \
  --resource-group myResourceGroup \
  --name placement-predictor \
  --image placement-predictor:latest \
  --dns-name-label placement-predictor \
  --ports 5000
```

## Production Checklist

### Security
- [ ] HTTPS enabled
- [ ] Security headers configured
- [ ] Input validation implemented
- [ ] Rate limiting configured
- [ ] Environment variables secured

### Performance
- [ ] Gunicorn with multiple workers
- [ ] Static file caching
- [ ] Database connection pooling (if applicable)
- [ ] CDN for static assets

### Monitoring
- [ ] Health check endpoints
- [ ] Logging configured
- [ ] Error tracking (Sentry, etc.)
- [ ] Performance monitoring

### Backup
- [ ] Model files backed up
- [ ] Database backups (if applicable)
- [ ] Configuration backups

## Environment Variables

```bash
# Production
FLASK_ENV=production
PORT=5000

# Optional
LOG_LEVEL=INFO
CORS_ORIGINS=https://yourdomain.com
```

## Health Checks

The application includes health check endpoints:

- **Health Check**: `GET /api/health`
- **Model Info**: `GET /api/model-info`

## Troubleshooting

### Common Issues

1. **Model not loading**:
   - Check if model files exist in `models/` directory
   - Verify file permissions

2. **Port already in use**:
   - Change port in app.py or use environment variable
   - Kill existing process: `lsof -ti:5000 | xargs kill -9`

3. **Memory issues**:
   - Reduce number of Gunicorn workers
   - Increase container memory limits

### Logs

- **Development**: Console output
- **Production**: `app.log` and `production.log` files
- **Docker**: `docker logs container-name`

## Performance Optimization

1. **Gunicorn Configuration**:
   ```bash
   gunicorn --workers 4 --worker-class gevent --timeout 120 wsgi:app
   ```

2. **Nginx Reverse Proxy** (optional):
   ```nginx
   server {
       listen 80;
       server_name yourdomain.com;
       
       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

## SSL/HTTPS Setup

### Let's Encrypt with Certbot

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

### Heroku

```bash
heroku certs:auto:enable
```

## Scaling

### Horizontal Scaling
- Load balancer with multiple instances
- Auto-scaling groups (AWS)
- Kubernetes deployment

### Vertical Scaling
- Increase container resources
- Optimize model inference
- Database optimization (if applicable) 