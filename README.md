# llm-compass

## Docker notes (on Windows)
### Setup
#### 1. Enable WSL2
- powershell (admin): `wsl --install` -> `wsl --update` -> `wsl --set-default-version 2`
- verify it's working: `wsl --version`

#### 2. Docker Desktop
- [Download](https://www.docker.com/products/docker-desktop/) and install Docker (Ensure "Use WSL 2 instead of Hyper-V" is checked in installer). 
- Reboot

#### 3. Launch Docker
Verify it's working in cmd / powershell: `docker --version`

### Commands
```powershell
# Build and start
docker-compose up --build

# Run Python commands in the container
docker-compose exec app python -m llm_compass.path.to.script

# Access PostgreSQL
docker-compose exec postgres psql -U admin -d database

# Stop everything:
docker-compose down

# Stop and **delete volumes** (fresh start)
docker-compose down -v
```

In postgres database use [SQL / PSQL commands](https://www.geeksforgeeks.org/postgresql/postgresql-psql-commands/)
