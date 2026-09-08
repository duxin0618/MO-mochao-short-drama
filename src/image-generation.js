import { imageProviderDefaults } from "./config.js";

const cleanBase=value=>String(value||"").trim().replace(/\/+$/g,"");
const imageError=(provider,response,data)=>data?.base_resp?.status_msg||data?.error?.message||data?.message||`${provider} 图片接口错误 ${response.status}`;

async function downloadImage(url){
  const parsed=new URL(String(url||""));
  const trustedBaidu=/\.bcebos\.com$/i.test(parsed.hostname);
  if(!trustedBaidu||!["http:","https:"].includes(parsed.protocol))throw new Error("千帆返回了无法验证的图片地址");
  const response=await fetch(parsed,{redirect:"follow",signal:AbortSignal.timeout(60000)});
  if(!response.ok)throw new Error(`下载生成图片失败（HTTP ${response.status}）`);
  const type=String(response.headers.get("content-type")||"").split(";")[0].toLowerCase();
  if(!type.startsWith("image/"))throw new Error("生图接口返回的地址不是图片");
  const bytes=Buffer.from(await response.arrayBuffer());
  if(!bytes.length)throw new Error("生图接口返回了空图片");
  if(bytes.length>30*1024*1024)throw new Error("生成图片超过30MB，已停止保存");
  return {bytes,mimeType:type};
}

export async function generateImage({providerId,apiKey,baseUrl,model,prompt}){
  const preset=imageProviderDefaults[providerId];
  if(!preset)throw new Error("不支持的文生图供应商");
  if(!apiKey)throw new Error("请先配置文生图 API Key");
  const base=cleanBase(baseUrl||preset.baseUrl),chosenModel=String(model||preset.model).trim();
  if(!base||!chosenModel)throw new Error("请填写文生图 Base URL 和模型名称");
  const endpoint=providerId==="minimax"?`${base}/image_generation`:`${base}/images/generations`;
  const safePrompt=providerId==="baidu_qwen"?String(prompt||"").slice(0,800):prompt;
  const payload=providerId==="minimax"
    ? {model:chosenModel,prompt,aspect_ratio:"2:3",response_format:"base64",n:1,prompt_optimizer:true}
    : {model:chosenModel,prompt:safePrompt,n:1,size:"768x1024",prompt_extend:true,watermark:false};
  const response=await fetch(endpoint,{method:"POST",headers:{Authorization:`Bearer ${apiKey}`,"Content-Type":"application/json"},body:JSON.stringify(payload),signal:AbortSignal.timeout(180000)});
  const data=await response.json().catch(()=>({}));
  if(!response.ok||data?.base_resp?.status_code)throw new Error(imageError(preset.label,response,data));
  const encoded=providerId==="minimax"?data?.data?.image_base64?.[0]:data?.data?.[0]?.b64_json;
  if(encoded){const bytes=Buffer.from(String(encoded),"base64");if(!bytes.length)throw new Error(`${preset.label} 返回了空图片`);return {bytes,mimeType:"image/jpeg"};}
  const url=data?.data?.[0]?.url||data?.data?.image_url?.[0];
  if(url)return downloadImage(url);
  throw new Error(`${preset.label} 未返回可用的图片数据`);
}
