import 'dotenv/config'
import express, { Express, Request, Response } from "express"
import { MongoClient } from 'mongodb'
import { callAgent } from './agent'
import cors from  'cors'

const app: Express = express();

app.use(cors());
app.use(express.json());

const client = new MongoClient(process.env.MONGODB_ATLAS_URI  as string)


 async function startServer() {
    try {
        await client.connect();
        console.log('Connected to MongoDB');

        app.get('/', async (req: Request, res: Response) => {
            res.send('Hello World');
        });
        app.post('/chat', async (req: Request, res: Response) => {
            const { message } = req.body;
            const threadId = Date.now().toString();
            try {
               const response = await callAgent(client,message,threadId);
               res.json({ threadId, response });
            } catch (error) {
                console.error('Error in chat agent:', error);
                res.status(500).json({ error: 'Internal server error' });
            }
        });
        app.post('/chat/:threadId', async (req: Request, res: Response) => {
            const threadId = String(req.params.threadId);
            const { message } = req.body;
            try {
               const response = await callAgent(client, message, threadId);
               res.json({ threadId, response });
            } catch (error) {
                console.error('Error in get chat history agent:', error);
                res.status(500).json({ error: 'Internal server error' });
            }
        });
        const PORT = process.env.PORT || 3000;
        app.listen(PORT, () => {
            console.log(`Server is running on port ${PORT}`);
        });
    } catch (error) {
        console.error('Error starting server:', error);
        process.exit(1);
    }
}

startServer();