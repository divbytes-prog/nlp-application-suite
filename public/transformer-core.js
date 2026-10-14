export const MODELS = {
  sentiment: {id: 'Xenova/twitter-roberta-base-sentiment-latest', revision: 'f3ec4d0925f90c3ca7ee7814f52d6ee7cf180445', labels: ['negative', 'neutral', 'positive']},
  emotion: {id: 'TrumpMcDonaldz/bhadresh-savani-distilbert-base-uncased-emotion-ONNX', revision: '17dd05bd2206a495d9b796e4103cb0089cdc59be', model_file_name: 'decoder_model_merged', labels: ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']},
};
export function modelText(text) {
  return text.replace(/https?:\/\/\S+/g, 'http').replace(/@[\p{L}\p{N}_]+/gu, '@user');
}
export function convert(rows, task, threshold = .60, precision = 'quantized') {
  const labels = MODELS[task].labels;
  const scores = {};
  for (const row of rows) {
    let label = row.label.toLowerCase();
    if (label.startsWith('label_')) label = labels[Number(label.slice(6))];
    if (!labels.includes(label) || !Number.isFinite(row.score)) throw new Error('Unexpected model output.');
    scores[label] = row.score;
  }
  if (Object.keys(scores).length !== labels.length) throw new Error('Incomplete model output.');
  const label = labels.reduce((a, b) => scores[a] >= scores[b] ? a : b);
  return {label, scores, confidence: scores[label], uncertain: scores[label] < threshold,
    confidence_kind: `uncalibrated ${precision === 'full' ? 'ONNX' : 'quantized ONNX'} softmax`, evidence: [],
    explanation_note: 'Token deletion shows sensitivity, not causal attribution. Browser ONNX scores can differ from Python scores.'};
}
const EMOJIS = {joy:['😊','😄','🎉'],love:['❤️','🥰','😍'],sadness:['😢','💙','🫂'],anger:['😤','😡','💢'],fear:['😟','😰','🫂'],surprise:['😮','🤯','✨'],positive:['😊','✨','👍'],negative:['😔','💙','🫂'],neutral:['💬','📝','🤔']};
export function recommend(text, sentiment, emotion) {
  const tokens = new Set((text.normalize('NFKC').toLowerCase().replaceAll('’', "'").match(/[\p{L}\p{N}_]+(?:'[\p{L}\p{N}_]+)?/gu) || []));
  const keywords = {birthday:'🎂',coffee:'☕',travel:'✈️',exam:'📚',win:'🏆'};
  return [...new Set([...Object.entries(keywords).filter(([word]) => tokens.has(word)).map(([,emoji]) => emoji), ...(EMOJIS[emotion] || EMOJIS[sentiment] || EMOJIS.neutral)])].slice(0,3);
}
export async function predict(pipe, text, task, options = {}) {
  const prepared = modelText(text);
  const infer = async value => convert(await pipe(value, {topk: null, truncation: true, max_length: 256}), task, options.threshold, options.precision);
  const result = await infer(prepared);
  if (options.explain) {
    const words = prepared.split(/\s+/).filter(Boolean);
    for (let i = 0; i < Math.min(16, words.length); i++) {
      const altered = await infer(words.filter((_,index) => index !== i).join(' ') || ' ');
      result.evidence.push({token: words[i], index: i, contribution: Number((result.confidence - altered.scores[result.label]).toFixed(5)), rule: 'token deletion sensitivity'});
      options.onProgress?.(`Explaining ${task}: ${i+1}/${Math.min(16,words.length)} words`);
    }
  }
  return result;
}
export async function analyzeModels(pipes, text, options = {}) {
  const started = performance.now();
  const sentiment = await predict(pipes.sentiment, text, 'sentiment', options);
  const emotion = await predict(pipes.emotion, text, 'emotion', options);
  const warnings = ['Text emotion is an uncertain inference, not a mental-health diagnosis.', 'These ONNX conversions are not measured by the Python benchmark displayed in Evaluation.'];
  if (sentiment.uncertain || emotion.uncertain) warnings.push('Low model confidence: treat this result as uncertain.');
  if (['kill myself','end my life','suicide'].some(phrase => text.toLowerCase().includes(phrase))) warnings.push('If this text reflects immediate danger, contact someone you trust or local emergency support. This tool cannot assess risk.');
  return {engine:'browser',model:`${MODELS.sentiment.id} + ${MODELS.emotion.id}`,sentiment,emotion,
    mood: ({sadness:'sad',joy:'happy',love:'affectionate',anger:'angry',fear:'anxious',surprise:'surprised'})[emotion.label],
    emojis: recommend(text,sentiment.label,emotion.label),warnings,latency_ms:Math.round((performance.now()-started)*100)/100};
}
