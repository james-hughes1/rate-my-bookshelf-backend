# Rate my bookshelf backend

# How to use

This backend is containerised in Docker and run on Google Cloud Run, with the following commands:

`docker build -t bookshelf-backend .`

`docker run -e GEMINI_API_KEY=123 -p 8080:8080 bookshelf-backend`

`docker tag bookshelf-backend:latest europe-west1-docker.pkg.dev/<project-id>/docker-repo/bookshelf-backend:latest`

`docker push europe-west1-docker.pkg.dev/<project-id>/docker-repo/bookshelf-backend:latest`