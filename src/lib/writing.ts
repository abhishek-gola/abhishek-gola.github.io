import { getCollection } from 'astro:content';
import { articles, profile } from '../data/site';

export async function getPublishedNotes() {
  return getCollection('writing', ({ data }) => !data.draft);
}

export async function getPublishedWriting() {
  const notes = await getPublishedNotes();
  return [
    ...articles.map(article => ({
      title: article.title,
      description: article.summary,
      date: new Date(`${article.date}T00:00:00Z`),
      topic: article.topic,
      authors: article.authors,
      publisher: article.publisher,
      url: article.url,
      external: true,
    })),
    ...notes.map(entry => ({
      title: entry.data.title,
      description: entry.data.description,
      date: entry.data.date,
      topic: entry.data.topic,
      authors: entry.data.authors.join(' · '),
      publisher: 'On this site',
      url: `/writing/${entry.id}/`,
      external: false,
    })),
  ].sort((a, b) => b.date.getTime() - a.date.getTime());
}

export function articleAuthor(name: string) {
  return { '@type': 'Person', name, ...(name === profile.name ? { url: profile.github } : {}) };
}
