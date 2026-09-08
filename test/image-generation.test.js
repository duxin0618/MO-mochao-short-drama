import test from "node:test";
import assert from "node:assert/strict";
import { generateImage } from "../src/image-generation.js";

test("百度千帆千问生图使用独立接口并读取返回图片",async()=>{
  const original=globalThis.fetch,calls=[];
  globalThis.fetch=async(url,options={})=>{
    calls.push({url:String(url),options});
    if(calls.length===1)return new Response(JSON.stringify({data:[{url:"http://qianfan-img-gen.bj.bcebos.com/qwen-image/result.png"}]}),{status:200,headers:{"content-type":"application/json"}});
    return new Response(Buffer.from([1,2,3]),{status:200,headers:{"content-type":"image/png"}});
  };
  try{
    const result=await generateImage({providerId:"baidu_qwen",apiKey:"test-key",baseUrl:"https://qianfan.baidubce.com/v2",model:"qwen-image",prompt:"人".repeat(900)});
    const body=JSON.parse(calls[0].options.body);
    assert.equal(calls[0].url,"https://qianfan.baidubce.com/v2/images/generations");
    assert.equal(calls[0].options.headers.Authorization,"Bearer test-key");
    assert.equal(body.prompt.length,800);
    assert.equal(body.n,1);
    assert.equal(result.mimeType,"image/png");
    assert.deepEqual([...result.bytes],[1,2,3]);
  }finally{globalThis.fetch=original;}
});
test("MiniMax 生图继续兼容 base64 响应",async()=>{
  const original=globalThis.fetch;
  globalThis.fetch=async()=>new Response(JSON.stringify({data:{image_base64:[Buffer.from("image").toString("base64")]}}),{status:200,headers:{"content-type":"application/json"}});
  try{
    const result=await generateImage({providerId:"minimax",apiKey:"test-key",baseUrl:"https://api.minimaxi.com/v1",model:"image-01",prompt:"人物"});
    assert.equal(result.bytes.toString(),"image");
  }finally{globalThis.fetch=original;}
});
