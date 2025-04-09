FROM nginx:alpine

# Install curl for healthcheck
RUN apk add --no-cache curl

# Copy custom nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Create log directory
RUN mkdir -p /var/log/nginx

# Expose port
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"] 