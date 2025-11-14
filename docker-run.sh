#!/bin/bash
# Helper script for running BrainToText2025 in Docker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_usage() {
    cat << EOF
Usage: ./docker-run.sh [command]

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
  ./docker-run.sh build
  ./docker-run.sh run-base
  ./docker-run.sh jupyter-lm
  ./docker-run.sh shell-base
EOF
}

check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed${NC}"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        echo -e "${RED}Error: Docker Compose is not installed${NC}"
        exit 1
    fi
}

# Main script
check_docker

case "${1:-help}" in
    build)
        echo -e "${GREEN}Building all Docker images...${NC}"
        docker-compose build
        echo -e "${GREEN}✓ Build complete${NC}"
        ;;
    
    build-base)
        echo -e "${GREEN}Building base environment...${NC}"
        docker-compose build braintotext-base
        echo -e "${GREEN}✓ Build complete${NC}"
        ;;
    
    build-lm)
        echo -e "${GREEN}Building language model environment...${NC}"
        docker-compose build braintotext-lm
        echo -e "${GREEN}✓ Build complete${NC}"
        ;;
    
    run-base)
        echo -e "${GREEN}Running pipeline with base environment...${NC}"
        docker-compose up braintotext-base
        ;;
    
    run-lm)
        echo -e "${GREEN}Running pipeline with language model environment...${NC}"
        docker-compose up braintotext-lm
        ;;
    
    jupyter-base)
        echo -e "${GREEN}Starting Jupyter Lab (base environment)...${NC}"
        echo -e "${YELLOW}Access at: http://localhost:8890${NC}"
        docker-compose up jupyter-base
        ;;
    
    jupyter-lm)
        echo -e "${GREEN}Starting Jupyter Lab (LM environment)...${NC}"
        echo -e "${YELLOW}Access at: http://localhost:8891${NC}"
        docker-compose up jupyter-lm
        ;;
    
    shell-base)
        echo -e "${GREEN}Opening shell in base environment...${NC}"
        docker-compose run --rm braintotext-base bash
        ;;
    
    shell-lm)
        echo -e "${GREEN}Opening shell in LM environment...${NC}"
        docker-compose run --rm braintotext-lm bash
        ;;
    
    viz-base)
        echo -e "${GREEN}Starting Kedro Viz (base environment)...${NC}"
        echo -e "${YELLOW}Access at: http://localhost:4141${NC}"
        docker-compose run --rm -p 4141:4141 braintotext-base kedro viz --host 0.0.0.0
        ;;
    
    viz-lm)
        echo -e "${GREEN}Starting Kedro Viz (LM environment)...${NC}"
        echo -e "${YELLOW}Access at: http://localhost:4142${NC}"
        docker-compose run --rm -p 4142:4141 braintotext-lm kedro viz --host 0.0.0.0
        ;;
    
    test)
        echo -e "${GREEN}Running tests...${NC}"
        docker-compose run --rm braintotext-base pytest
        ;;
    
    clean)
        echo -e "${YELLOW}Stopping and removing containers...${NC}"
        docker-compose down
        echo -e "${GREEN}✓ Cleanup complete${NC}"
        ;;
    
    clean-all)
        echo -e "${YELLOW}Removing containers, images, and volumes...${NC}"
        echo -e "${RED}Warning: This will delete all data in Docker volumes!${NC}"
        read -p "Are you sure? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            docker-compose down -v --rmi all
            echo -e "${GREEN}✓ Complete cleanup done${NC}"
        else
            echo -e "${YELLOW}Cleanup cancelled${NC}"
        fi
        ;;
    
    logs-base)
        docker-compose logs -f braintotext-base
        ;;
    
    logs-lm)
        docker-compose logs -f braintotext-lm
        ;;
    
    help|--help|-h)
        print_usage
        ;;
    
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo
        print_usage
        exit 1
        ;;
esac
