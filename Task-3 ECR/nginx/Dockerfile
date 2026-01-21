# Base NGINX image
FROM nginx:latest

# Remove default config
RUN rm /etc/nginx/conf.d/default.conf

# Copy our custom config
COPY default.conf /etc/nginx/conf.d/

# Expose port 80
EXPOSE 80

# Use default command
CMD ["nginx", "-g", "daemon off;"]
