import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MongoClient } from 'mongodb';
import { StructuredOutputParser } from "@langchain/core/output_parsers";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import dotenv from 'dotenv';
import { z } from 'zod';

dotenv.config();

const client = new MongoClient(process.env.MONGODB_ATLAS_URI as string);

const openRouterConfig = {
    baseURL: "https://openrouter.ai/api/v1",
    defaultHeaders: {
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "ecommerce-chat-helper",
    },
};

const llm = new ChatOpenAI({
    model: "google/gemini-2.5-flash",
    apiKey: process.env.OPENROUTER_API_KEY as string,
    temperature: 0.7,
    configuration: openRouterConfig,
});

const embeddings = new OpenAIEmbeddings({
    model: "openai/text-embedding-3-small",
    apiKey: process.env.OPENROUTER_API_KEY as string,
    dimensions: 1536,
    configuration: openRouterConfig,
});

const itemSchema = z.object({
    name: z.string(),
    description: z.string(),
    brand: z.string(),
    manufacturer_address: z.object({
        street: z.string(),
        city: z.string(),
        state: z.string(),
        zip: z.string(),
        country: z.string(),
    }),
    prices: z.object({
        full_price: z.number(),
        sale_price: z.number(),
        discount: z.number(),
    }),
    categories: z.array(z.string()),
    user_reviews: z.array(z.object({
        review_date: z.string(),
        rating: z.number(),
        comment: z.string(),
    })),
    notes: z.string()
});

type Item = z.infer<typeof itemSchema>;

const parser = StructuredOutputParser.fromZodSchema(z.array(itemSchema));

async function setDatabaseAndCollection() : Promise<void> {
    console.log('Setting database and collection');
    const db = client.db('inventory_database');
    const collection = await db.listCollections({ name: 'items' } as any).toArray();
    if (collection.length === 0) {
        await db.createCollection('items');
        console.log('Created items collection in inventory_database database');
    } else {
        console.log('Items collection already exists in inventory_database database');
    }
}

async function createVectorSearchIndex(): Promise<void> {
    try {
        const db = client.db('inventory_database');
        const collection = db.collection('items');
        console.log('Creating vector search index...');
        try {
            await collection.dropSearchIndex('vector_index');
        } catch {
            // Index may not exist yet
        }
        const vectorSearchIndex = {
            name: 'vector_index',
            type: 'vectorSearch',
            definition: {
                fields: [
                    {
                        type: 'vector',
                        path: 'embedding',
                        numDimensions: 1536,
                        similarity: 'cosine',
                    },
                ],
            },
        };
        await collection.createSearchIndex(vectorSearchIndex);
        console.log('Vector search index created');
    } catch (error) {
        console.error('Failed to create vector search index:', error);
    }
}

async function generateSyntheticData(): Promise<Item[]> {
    const prompt = `You are a helpful assistant that generates 
     furniture store item data. Generate 10 furniture store items. Each record
     should include the following fields: item_id, item_name, item_description,
     brand, manufacturer_address, prices, categories, user_reviews, notes. 
     Ensure variety in the data and realistic values.
     ${parser.getFormatInstructions()} `

     console.log('Generating synthetic data...');
     const response = await llm.invoke(prompt);
     const data = await parser.parse(response.content as string);
     console.log('Generated data:', data);
     return data;
}

async function createItemSummary(item: Item): Promise<string> {
    return new Promise((resolve) => {
        const manufacturerDetails = `Made in ${item.manufacturer_address.country}`;
        const categories = item.categories.join(', ');
        const userReviews = item.user_reviews.map(review => `Rated ${review.rating} on ${review.review_date}: ${review.comment}`).join('\n');
        const basicInfo = `${item.name} ${item.description} from the brand ${item.brand}`;
        const price = `At full price it consts: ${item.prices.full_price} USD,
        On sale it costs: ${item.prices.sale_price} USD`;
        const notes = item.notes;
        const summary = `${basicInfo}. Manufacturer: ${manufacturerDetails}.
        Categories: ${categories}. User Reviews: ${userReviews}. Price: ${price}. Notes: ${notes}.`;
        resolve(summary);
    });
}

async function seedDatabase(): Promise<void> {
    try {
        await client.connect();
        await client.db('admin').command({ ping: 1 });
        console.log("Pinged your deployment. You successfully connected to MongoDB!");
        await setDatabaseAndCollection();
        await createVectorSearchIndex();

        const db = client.db('inventory_database');
        const collection = db.collection('items');
        await collection.deleteMany({});
        console.log('Deleted all documents from items collection');

        const syntheticData = await generateSyntheticData();
        const recordsWithSummaries = await Promise.all(
            syntheticData.map(async (item) => ({
                pageContent: await createItemSummary(item),
                metadata: { ...item },
            }))
        );

        for (const record of recordsWithSummaries) {
            await MongoDBAtlasVectorSearch.fromDocuments(
                [record],
                embeddings,
                {
                    collection,
                    indexName: 'vector_index',
                    textKey: 'embedding_text',
                    embeddingKey: 'embedding',
                }
            );
            console.log('Successfully processed & saved record:', record.metadata.name);
        }

        console.log('Database seeded successfully completed');
    } catch (error) {
        console.error('Error seeding database:', error);
    } finally {
        await client.close();
    }
}

seedDatabase().catch(console.error);