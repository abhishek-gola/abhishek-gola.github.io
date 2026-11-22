import { speakerBio } from '../data/speaking';
import { profile } from '../data/site';
export function GET() {
  return new Response(`${speakerBio}\n\nContact: ${profile.email}\nPortfolio: https://abhishek-gola.github.io/\nLocation: Bengaluru, India\nSpeaking outside Bengaluru: travel and accommodation support required.\n`, {headers:{'Content-Type':'text/plain; charset=utf-8'}});
}
