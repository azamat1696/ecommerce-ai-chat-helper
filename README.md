# Ecommerce AI Chat Helper

An AI-powered furniture store assistant. Customers ask natural-language questions about inventory; a LangGraph agent searches a MongoDB Atlas vector index and answers with product details.

## Features

- **Conversational shopping assistant** — multi-turn chat with thread persistence
- **Semantic product search** — OpenRouter embeddings + MongoDB Atlas Vector Search
- **Synthetic inventory seeding** — generates furniture items and stores embeddings
- **React storefront UI** — ecommerce layout with an embedded chat widget

## Architecture

```
client (React)  →  POST /chat  →  server (Express + LangGraph agent)
                                      ↓
                               MongoDB Atlas
                         (items + vector_index + checkpoints)
                                      ↓
                               OpenRouter API
                    (chat: google/gemini-2.5-flash
                     embeddings: openai/text-embedding-3-small)
```

| Layer | Role |
| --- | --- |
| `client/` | React storefront and chat widget |
| `server/` | Express API, LangGraph agent, seed script |
| MongoDB Atlas | Product inventory, vector search, conversation checkpoints |
| OpenRouter | LLM + embedding provider |

## Tech stack

- **Backend:** Node.js, Express, TypeScript, LangChain, LangGraph
- **Frontend:** React 19
- **Database:** MongoDB Atlas (Vector Search)
- **AI:** OpenRouter (`ChatOpenAI` / `OpenAIEmbeddings`)

## Prerequisites

- Node.js 18+
- A [MongoDB Atlas](https://cloud.mongodb.com) cluster with Network Access configured for your IP
- An [OpenRouter](https://openrouter.ai/keys) **API key** (not a management/provisioning key)

## Setup

### 1. Clone and install

```bash
git clone https://github.com/azamat1696/ecommerce-ai-chat-helper.git
cd ecommerce-ai-chat-helper

cd server && npm install && cd ..
cd client && npm install && cd ..
```

### 2. Configure environment

```bash
cp server/.env.example server/.env
```

Fill in `server/.env`:

```env
PORT=8000
MONGODB_ATLAS_URI=mongodb+srv://<user>:<password>@<cluster>.mongodb.net
MONGODB_USERNAME=
MONGODB_PASSWORD=
MONGODB_URI=
OPENROUTER_API_KEY=sk-or-v1-...
```

Never commit `server/.env`. Only `server/.env.example` belongs in git.

### 3. Seed the database

Creates `inventory_database.items`, a `vector_index` (1536 dimensions), and sample furniture products with embeddings:

```bash
cd server
npm run seed
```

### 4. Run the app

Terminal 1 — API (default port from `.env`, e.g. `8000`):

```bash
cd server
npm run dev
```

Terminal 2 — UI:

```bash
cd client
npm start
```

Open the storefront (usually `http://localhost:3000`) and use the chat widget.

## API

### Start a chat

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Do you have any dining tables?"}'
```

Response includes a `threadId` for follow-up turns.

### Continue a conversation

```bash
curl -X POST http://localhost:8000/chat/<threadId> \
  -H "Content-Type: application/json" \
  -d '{"message": "What colors are available?"}'
```

## How the agent works

1. User message enters a LangGraph workflow.
2. The model decides whether to call `item_lookup`.
3. `item_lookup` runs vector similarity search on `items` (falls back to regex text search if needed).
4. The model answers using retrieved product data.
5. Conversation state is checkpointed in MongoDB per `threadId`.

## Project structure

```
ecommerce-chat-helper/
├── client/                 # React storefront + ChatWidget
├── server/
│   ├── agent.ts            # LangGraph agent + inventory tool
│   ├── index.ts            # Express API
│   ├── seed-database.ts    # Synthetic data + embeddings
│   ├── .env.example
│   └── package.json
├── .gitignore
└── README.md
```

## Troubleshooting

| Issue | Fix |
| --- | --- |
| `tlsv1 alert internal error` on Mongo connect | Add your IP in Atlas → **Network Access** (or temporarily `0.0.0.0/0`) |
| OpenRouter `401 User not found` | Use a normal API key, not a management/provisioning key |
| Empty / wrong search results | Re-run `npm run seed`; confirm Atlas index `vector_index` exists on `embedding` (1536 dims, cosine) |
| Client can't reach API | Ensure server `PORT` matches the URL in `ChatWidget` (default `http://localhost:8000`) |

## License

ISC
