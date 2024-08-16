# Dockerfile

# Use an official Node.js runtime as a parent image
FROM node:18-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy package.json and yarn.lock to install dependencies
COPY package.json yarn.lock ./

RUN yarn install

# Copy the rest of the application code
COPY . .

# Expose the port that Vite runs on (default is 5173)
EXPOSE 5173

# Command to run the Vite development server
CMD ["yarn", "dev"]
