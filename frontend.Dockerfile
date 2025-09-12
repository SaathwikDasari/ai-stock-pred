FROM node:20-bullseye

WORKDIR /app

COPY FRONTEND/ai-stock-pred-next/package*.json ./

RUN npm install

COPY FRONTEND/ai-stock-pred-next .

EXPOSE 3000

CMD ["npm", "run", "dev"]
