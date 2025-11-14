# Docker Setup for BrainToText2025

This document provides comprehensive instructions for running the BrainToText2025 project using Docker. The project uses two separate Docker environments to handle different computational requirements:

- **b2txt25 (base)**: Data processing environment with Kedro, PySpark, and scikit-learn
- **b2txt25_lm (language model)**: Deep learning environment with PyTorch and transformer models

## Prerequisites

- Docker (version 20.10 or later)
- Docker Compose (version 2.0 or later)
- At least 8GB of free disk space
- (Optional) NVIDIA Docker for GPU support

### Installing Docker

#### Windows
1. Download and install [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
2. Ensure WSL 2 backend is enabled
3. Start Docker Desktop

#### Linux
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### macOS
1. Download and install [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
2. Start Docker Desktop

## Quick Start

### 1. Build Docker Images

Build both environments:

```bash
# Build base environment (b2txt25)
docker-compose build braintotext-base

# Build language model environment (b2txt25_lm)
docker-compose build braintotext-lm
```

Or build both at once:

```bash
docker-compose build
```

### 2. Running the Pipeline

#### Run with Base Environment
```bash
docker-compose up braintotext-base
```

#### Run with Language Model Environment
```bash
docker-compose up braintotext-lm
```

### 3. Interactive Development with Jupyter

#### Launch Jupyter Lab (Base Environment)
```bash
docker-compose up jupyter-base
```
Access at: http://localhost:8890

#### Launch Jupyter Lab (LM Environment)
```bash
docker-compose up jupyter-lm
```
Access at: http://localhost:8891

## Detailed Usage

### Working with Individual Containers

#### Execute Commands in Running Container
```bash
# Base environment
docker exec -it braintotext-base bash

# Language model environment
docker exec -it braintotext-lm bash
```

#### Run Specific Kedro Pipeline
```bash
# Base environment
docker-compose run --rm braintotext-base kedro run --pipeline=data_processing

# LM environment
docker-compose run --rm braintotext-lm kedro run --pipeline=model_training
```

#### Run Tests
```bash
docker-compose run --rm braintotext-base pytest
```

#### Access Kedro Viz
```bash
# Base environment
docker-compose run --rm -p 4141:4141 braintotext-base kedro viz --host 0.0.0.0

# LM environment
docker-compose run --rm -p 4142:4141 braintotext-lm kedro viz --host 0.0.0.0
```

### Data Management

#### Mounting Data Volumes
The `docker-compose.yml` is configured to mount the following directories:
- `./data` → `/app/data` (shared across both environments)
- `./models` → `/app/models` (for LM environment)
- `./conf/local` → `/app/conf/local` (local configuration)
- `./notebooks` → `/app/notebooks` (Jupyter notebooks)

#### Adding Data
1. Place your data files in the `data/01_raw` directory on your host machine
2. The data will be automatically available in the container

```bash
# Example: Copy data to container
cp /path/to/your/data.csv ./data/01_raw/
```

### Environment Configuration

#### Setting Environment Variables
Edit `docker-compose.yml` to add environment variables:

```yaml
environment:
  - KEDRO_ENV=local
  - CUSTOM_VAR=value
```

#### Using Local Configuration
Place local configuration files in `conf/local/`:
- `credentials.yml` - for secrets and credentials
- `local.yml` - for local parameter overrides

These files are git-ignored and mounted as volumes.

## GPU Support

### Enabling GPU for Language Model Environment

#### Prerequisites
- NVIDIA GPU
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

#### Modify Dockerfile.lm
Replace the CPU PyTorch installation with GPU version:

```dockerfile
# Replace this line:
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

# With this (for CUDA 11.8):
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu118
```

#### Enable GPU in docker-compose.yml
Uncomment the GPU section in `docker-compose.yml`:

```yaml
braintotext-lm:
  # ... other config ...
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

#### Verify GPU Access
```bash
docker-compose run --rm braintotext-lm python -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting

### Build Issues

#### Out of Memory During Build
Increase Docker memory allocation:
- **Docker Desktop**: Settings → Resources → Memory (set to at least 8GB)

#### Slow Build Times
Use build cache and buildkit:
```bash
DOCKER_BUILDKIT=1 docker-compose build
```

### Runtime Issues

#### Permission Denied Errors
On Linux, you may need to adjust file permissions:
```bash
sudo chown -R $USER:$USER ./data ./models ./logs
```

#### Port Already in Use
Change ports in `docker-compose.yml`:
```yaml
ports:
  - "4143:4141"  # Use different host port
```

#### Container Exits Immediately
Check logs:
```bash
docker-compose logs braintotext-base
```

### Data Access Issues

#### Data Not Visible in Container
Verify volume mounts:
```bash
docker-compose run --rm braintotext-base ls -la /app/data
```

## Best Practices

### Development Workflow
1. Use Jupyter containers for interactive development
2. Mount source code as volume to enable hot-reloading
3. Use `.dockerignore` to optimize build times
4. Keep data and models outside the image (use volumes)

### Production Considerations
1. Use multi-stage builds for smaller images
2. Pin all dependency versions
3. Set resource limits in `docker-compose.yml`
4. Use Docker secrets for credentials
5. Implement health checks

### Resource Management
```yaml
services:
  braintotext-lm:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 16G
        reservations:
          cpus: '2'
          memory: 8G
```

## Customization

### Adding Python Dependencies

#### Base Environment
Edit `requirements.txt` and rebuild:
```bash
docker-compose build braintotext-base
```

#### LM Environment Only
Edit `Dockerfile.lm` to add LM-specific packages:
```dockerfile
RUN pip install --no-cache-dir \
    your-package==1.0.0
```

### Multiple Pipelines
You can define additional services in `docker-compose.yml` for different pipelines:

```yaml
services:
  preprocessing:
    extends: braintotext-base
    command: kedro run --pipeline=preprocessing
    
  training:
    extends: braintotext-lm
    command: kedro run --pipeline=training
```

## Cleaning Up

### Remove Containers
```bash
docker-compose down
```

### Remove Images
```bash
docker-compose down --rmi all
```

### Remove Volumes (⚠️ Deletes Data)
```bash
docker-compose down -v
```

### Complete Cleanup
```bash
# Remove everything including cached layers
docker-compose down -v --rmi all
docker system prune -a
```

## Port Reference

| Service | Host Port | Container Port | Purpose |
|---------|-----------|----------------|---------|
| braintotext-base | 4141 | 4141 | Kedro Viz |
| braintotext-lm | 4142 | 4141 | Kedro Viz |
| jupyter-base | 8890 | 8888 | Jupyter Lab |
| jupyter-lm | 8891 | 8888 | Jupyter Lab |

## Support

For issues specific to the Docker setup, check:
1. Docker logs: `docker-compose logs -f [service-name]`
2. Container status: `docker-compose ps`
3. Resource usage: `docker stats`

For Kedro-specific issues, see the [main README.md](README.md).
