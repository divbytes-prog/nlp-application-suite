import {pipeline,env} from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js';
import {MODELS,analyzeModels} from './transformer-core.js';
env.allowLocalModels=false;
env.backends.onnx.wasm.numThreads=1;
let pipelines={},loadedPrecision;
async function loadModels(id,precision) {
  if (loadedPrecision!==precision) {
    for (const pipe of Object.values(pipelines)) await pipe.dispose();
    pipelines={}; loadedPrecision=precision;
  }
  for (const [task,spec] of Object.entries(MODELS)) {
    if (pipelines[task]) continue;
    const progress_callback=progress=>{
      const percent=Number.isFinite(progress.progress)?` ${Math.round(progress.progress)}%`:'';
      self.postMessage({id,status:'progress',message:`Loading ${task} model${percent}. First use downloads model files; later runs use the browser cache.`});
    };
    pipelines[task]=await pipeline('text-classification',spec.id,{revision:spec.revision,model_file_name:spec.model_file_name,quantized:precision!=='full',progress_callback});
  }
}
let queue=Promise.resolve();
self.addEventListener('message',event=>{
  const {id,text,texts,explain=false,precision='quantized',threshold=.60}=event.data;
  queue=queue.then(async()=>{
    try {
      await loadModels(id,precision);
      const options={explain,precision,threshold,onProgress:message=>self.postMessage({id,status:'progress',message})};
      if (texts) {
        const results=[];
        for (let i=0;i<texts.length;i++) {
          self.postMessage({id,status:'progress',message:`Analyzing row ${i+1} of ${texts.length} with both transformers…`});
          results.push(await analyzeModels(pipelines,texts[i],{...options,explain:false}));
        }
        self.postMessage({id,results});
      } else self.postMessage({id,result:await analyzeModels(pipelines,text,options)});
    } catch(error) {
      self.postMessage({id,error:`Transformer inference failed: ${error.message || 'model unavailable'}. Check network access and available device memory, then retry.`});
    }
  });
});
