# Helper script for running BrainToText2025 in Docker (PowerShell)

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

function Print-Usage {
    Write-Host @"
Usage: .\docker-run.ps1 [command]

Commands:
  build              Build both Docker images
  build-base         Build base environment only
  build-lm           Build language model environment only
  
  run-base           Run pipeline with base environment
  run-lm             Run pipeline with language model environment
  
  jupyter-base       Start Jupyter Lab with base environment (port 8890)
  jupyter-lm         Start Jupyter Lab with LM environment (port 8891)
  
  shell-base         Open bash shell in base environment
  shell-lm           Open bash shell in LM environment
  
  viz-base           Start Kedro Viz for base environment (port 4141)
  viz-lm             Start Kedro Viz for LM environment (port 4142)
  
  test               Run tests in base environment
  
  clean              Stop and remove containers
  clean-all          Remove containers, images, and volumes
  
  logs-base          View logs for base environment
  logs-lm            View logs for LM environment
  
  help               Show this help message

Examples:
  .\docker-run.ps1 build
  .\docker-run.ps1 run-base
  .\docker-run.ps1 jupyter-lm
  .\docker-run.ps1 shell-base
"@
}

function Check-Docker {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Host "Error: Docker is not installed" -ForegroundColor Red
        exit 1
    }
    
    if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
        Write-Host "Error: Docker Compose is not installed" -ForegroundColor Red
        exit 1
    }
}

# Main script
Check-Docker

switch ($Command) {
    "build" {
        Write-Host "Building all Docker images..." -ForegroundColor Green
        docker-compose build
        Write-Host "✓ Build complete" -ForegroundColor Green
    }
    
    "build-base" {
        Write-Host "Building base environment..." -ForegroundColor Green
        docker-compose build braintotext-base
        Write-Host "✓ Build complete" -ForegroundColor Green
    }
    
    "build-lm" {
        Write-Host "Building language model environment..." -ForegroundColor Green
        docker-compose build braintotext-lm
        Write-Host "✓ Build complete" -ForegroundColor Green
    }
    
    "run-base" {
        Write-Host "Running pipeline with base environment..." -ForegroundColor Green
        docker-compose up braintotext-base
    }
    
    "run-lm" {
        Write-Host "Running pipeline with language model environment..." -ForegroundColor Green
        docker-compose up braintotext-lm
    }
    
    "jupyter-base" {
        Write-Host "Starting Jupyter Lab (base environment)..." -ForegroundColor Green
        Write-Host "Access at: http://localhost:8890" -ForegroundColor Yellow
        docker-compose up jupyter-base
    }
    
    "jupyter-lm" {
        Write-Host "Starting Jupyter Lab (LM environment)..." -ForegroundColor Green
        Write-Host "Access at: http://localhost:8891" -ForegroundColor Yellow
        docker-compose up jupyter-lm
    }
    
    "shell-base" {
        Write-Host "Opening shell in base environment..." -ForegroundColor Green
        docker-compose run --rm braintotext-base bash
    }
    
    "shell-lm" {
        Write-Host "Opening shell in LM environment..." -ForegroundColor Green
        docker-compose run --rm braintotext-lm bash
    }
    
    "viz-base" {
        Write-Host "Starting Kedro Viz (base environment)..." -ForegroundColor Green
        Write-Host "Access at: http://localhost:4141" -ForegroundColor Yellow
        docker-compose run --rm -p 4141:4141 braintotext-base kedro viz --host 0.0.0.0
    }
    
    "viz-lm" {
        Write-Host "Starting Kedro Viz (LM environment)..." -ForegroundColor Green
        Write-Host "Access at: http://localhost:4142" -ForegroundColor Yellow
        docker-compose run --rm -p 4142:4141 braintotext-lm kedro viz --host 0.0.0.0
    }
    
    "test" {
        Write-Host "Running tests..." -ForegroundColor Green
        docker-compose run --rm braintotext-base pytest
    }
    
    "clean" {
        Write-Host "Stopping and removing containers..." -ForegroundColor Yellow
        docker-compose down
        Write-Host "✓ Cleanup complete" -ForegroundColor Green
    }
    
    "clean-all" {
        Write-Host "Removing containers, images, and volumes..." -ForegroundColor Yellow
        Write-Host "Warning: This will delete all data in Docker volumes!" -ForegroundColor Red
        $confirmation = Read-Host "Are you sure? (y/N)"
        if ($confirmation -eq 'y' -or $confirmation -eq 'Y') {
            docker-compose down -v --rmi all
            Write-Host "✓ Complete cleanup done" -ForegroundColor Green
        } else {
            Write-Host "Cleanup cancelled" -ForegroundColor Yellow
        }
    }
    
    "logs-base" {
        docker-compose logs -f braintotext-base
    }
    
    "logs-lm" {
        docker-compose logs -f braintotext-lm
    }
    
    "help" {
        Print-Usage
    }
    
    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host ""
        Print-Usage
        exit 1
    }
}
