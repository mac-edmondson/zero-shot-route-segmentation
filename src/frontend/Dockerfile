# Build dependencies and compile the Next.js application.
FROM node:22-alpine AS build
WORKDIR /app

# Public Next.js variables are embedded into the client bundle at build time.
ARG NEXT_PUBLIC_API_BASE_URL
ARG NEXT_PUBLIC_USE_MOCK_API
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
ENV NEXT_PUBLIC_USE_MOCK_API=$NEXT_PUBLIC_USE_MOCK_API

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build && npm prune --omit=dev

# Keep the runtime image small and run as the non-root node user.
FROM node:22-alpine AS runtime
WORKDIR /app
ENV NODE_ENV=production

# Give the non-root runtime user ownership of copied application files.
COPY --from=build --chown=node:node /app/package*.json ./
COPY --from=build --chown=node:node /app/node_modules ./node_modules
COPY --from=build --chown=node:node /app/.next ./.next

USER node
EXPOSE 3000
CMD ["npm", "run", "start"]
