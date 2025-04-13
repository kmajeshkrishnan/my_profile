
FROM nginx:stable-alpine

# Install required tools
RUN apk add --no-cache openssl curl

# Make the script executable
COPY start-nginx.sh /start-nginx.sh
RUN chmod +x /start-nginx.sh

EXPOSE 80 443