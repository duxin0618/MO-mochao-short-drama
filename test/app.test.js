import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import mammoth from "mammoth";

process.env.APP_DATA_DIR = fs.mkdtempSync(path.join(os.tmpdir(), "mochao-test-"));
process.env.LLM_PROVIDER = "mock";
process.env.EMBEDDING_PROVIDER = "mock";
process.env.ALLOW_MOCK_WORKFLOWS = "1";
const { app } = await import("../src/app.js");
const { run, get } = await import("../src/db.js");
const { searchIdeaKnowledge, cleanupAutomaticKnowledge, seedAutomaticSources } = await import("../src/knowledge.js");
const { novelDescriptionIssues, novelPerspectiveIssue, splitNovelLongParagraphs, stripNovelRevisionHeading, isTransientTransportError, shortNovelExpansionInstruction } = await import("../src/llm.js");
const { embedTexts, embeddingBatchLimitError } = await import("../src/embeddings.js");
const { buildEpisodeNovelPrompt, buildOutlineDramaticBatchPrompt, buildSkillEpisodePrompt, buildStagePrompt } = await import("../src/prompts.js");
const { createWorkspaceBackup, restoreWorkspaceBackup, validateWorkspaceBackup } = await import("../src/backup.js");
const { detectInstallMode } = await import("../src/updater.js");
const { combatLinkView } = await import("../src/memory.js");
const server = app.listen(0, "127.0.0.1");
await new Promise(resolve => server.once("listening", resolve));
const base = `http://127.0.0.1:${server.address().port}`;

const request = async (url, options={}) => { const r=await fetch(base+url,{headers:{"content-type":"application/json"},...options}); const body=r.status===204?null:await r.json(); return {r,body}; };

test("建链视图只压缩连续打戏且保留普通事件",()=>{
  const event=(order,summary,overrides={})=>({order,scene_no:1,timeline_type:"main",event_type:"event",subject:"刘依灵",participants:["刘依灵","追兵"],object_text:"追兵",action:summary,result_text:"",summary,source_quote:`△${summary}`,...overrides});
  const input=[event(1,"追兵挥刀攻击刘依灵"),event(2,"刘依灵闪避追兵的刀锋"),event(3,"刘依灵反击并击倒追兵"),event(4,"刘依灵拒绝交出物资",{action:"拒绝",source_quote:"刘依灵：我不会交出物资"}),event(5,"系统提示任务完成",{event_type:"system_change",action:"提示"})];
  const view=combatLinkView(input);
  assert.equal(view.length,3);assert.deepEqual(view[0]._combat_orders,[1,2,3]);assert.equal(view[0].order,3);assert.equal(view[1],input[3]);assert.equal(view[2],input[4]);
});

test("故事状态共享重试只识别临时连接故障",()=>{
  assert.equal(isTransientTransportError(new Error("Connection error.")),true);
  assert.equal(isTransientTransportError(Object.assign(new Error("upstream unavailable"),{status:503})),true);
  assert.equal(isTransientTransportError(Object.assign(new Error("Unauthorized"),{status:401})),false);
  assert.equal(isTransientTransportError(new Error("输出因达到长度上限被截断")),false);
});

test("小说字数不足反馈按模板计算安全增量且保留扩写边界",()=>{
  const instruction=shortNovelExpansionInstruction("小说有效字符 1929，少于最低要求 2000",{minEffectiveCharacters:2000,maxEffectiveCharacters:4000,creativeMinCharacters:2000,creativeMaxCharacters:3000});
  assert.match(instruction,/至少净增加 371 字/);assert.match(instruction,/建议净增加 400–600 字/);assert.match(instruction,/达到 2300–2500 个纯汉字/);assert.match(instruction,/只扩充关键冲突回合、因果和必要细节，不增加新支线/);assert.match(instruction,/现场描写必须服务于处境、线索、压力、人物或既定选择/);assert.match(instruction,/不得改变既有事件结果/);
});

test("向量批量上限错误可以自适应识别",()=>{
  assert.deepEqual(embeddingBatchLimitError(new Error("InternalError.Algo.InvalidParameter: Value error, batch size is invalid, it should not be larger than 10.: input.contents")),{recognized:true,limit:10});
  assert.deepEqual(embeddingBatchLimitError(new Error("maximum batch size is 16")),{recognized:true,limit:16});
  assert.deepEqual(embeddingBatchLimitError(new Error("输入数量最多为 8 条")),{recognized:true,limit:8});
  assert.deepEqual(embeddingBatchLimitError(new Error("Invalid input: text cannot be empty")),{recognized:false,limit:null});
});

test("向量接口超限后自动分批并绑定API指纹",async()=>{
  const sizes=[];
  const embeddingServer=http.createServer(async(req,res)=>{
    let raw="";for await(const chunk of req)raw+=chunk;const body=JSON.parse(raw||"{}"),input=Array.isArray(body.input)?body.input:[],limited=req.headers.authorization==="Bearer key-a";sizes.push(input.length);
    if(limited&&input.length>3){res.writeHead(400,{"content-type":"application/json"});res.end(JSON.stringify({error:{message:"maximum batch size is 3",type:"invalid_request_error"}}));return;}
    res.writeHead(200,{"content-type":"application/json"});res.end(JSON.stringify({object:"list",model:body.model,data:input.map((value,index)=>({object:"embedding",index,embedding:[Number(String(value).replace(/\D/g,""))||0,1]})),usage:{prompt_tokens:input.length,total_tokens:input.length}}));
  });
  await new Promise(resolve=>embeddingServer.listen(0,"127.0.0.1",resolve));
  const baseUrl=`http://127.0.0.1:${embeddingServer.address().port}/v1`,values=Array.from({length:7},(_,index)=>`事件${index+1}`);
  try{
    const first=await embedTexts(values,{providerOverride:{id:"custom",baseUrl,model:"test-embedding",apiKey:"key-a"}});assert.equal(first.length,7);assert.deepEqual(sizes,[7,3,3,1]);
    sizes.length=0;await embedTexts(values,{providerOverride:{id:"custom",baseUrl,model:"test-embedding",apiKey:"key-a"}});assert.deepEqual(sizes,[3,3,1]);
    sizes.length=0;await embedTexts(values,{providerOverride:{id:"custom",baseUrl,model:"test-embedding",apiKey:"key-b"}});assert.deepEqual(sizes,[7]);
  }finally{await new Promise(resolve=>embeddingServer.close(resolve));}
});

test("健康检查和完整创作主链", async () => {
  const installFixture=fs.mkdtempSync(path.join(os.tmpdir(),"mochao-install-mode-"));fs.mkdirSync(path.join(installFixture,".git"));assert.equal(detectInstallMode(installFixture),"git");fs.rmSync(path.join(installFixture,".git"),{recursive:true});fs.writeFileSync(path.join(installFixture,"portable.json"),"{}");fs.mkdirSync(path.join(installFixture,"runtime"));fs.writeFileSync(path.join(installFixture,"runtime","node.exe"),"");assert.equal(detectInstallMode(installFixture),"portable");fs.rmSync(path.join(installFixture,"runtime"),{recursive:true});assert.equal(detectInstallMode(installFixture),"unsupported");fs.rmSync(installFixture,{recursive:true,force:true});
  const scopedConstraints=[
    {kind:"hard",category:"ending",description:"主角最终必须回家",episode_start:null,episode_end:null},
    {kind:"hard",category:"character",description:"苏婉清发现录音",episode_start:6,episode_end:6},
    {kind:"hard",category:"reveal",description:"反派身份不得揭晓",episode_start:7,episode_end:9}
  ];
  const scopedNovel=buildEpisodeNovelPrompt({narrative_person:"first",emotion_intensity:"strong"},scopedConstraints,[],{episode_no:6},[],{generationTarget:{}},{});assert.match(scopedNovel,/主角最终必须回家/);assert.match(scopedNovel,/苏婉清发现录音/);assert.doesNotMatch(scopedNovel,/反派身份不得揭晓/);assert.match(scopedNovel,/适用EP06/);
  const scopedBatch=buildOutlineDramaticBatchPrompt({total_episodes:10,tags:[]},scopedConstraints,[],[],[{episode_no:6}],[],[]);assert.match(scopedBatch,/苏婉清发现录音/);assert.doesNotMatch(scopedBatch,/反派身份不得揭晓/);
  const sinkingBatch=buildOutlineDramaticBatchPrompt({total_episodes:10,tags:[],story_mode:"miniprogram"},[],[],[],[{episode_no:1}],[],[]);assert.match(sinkingBatch,/软肋/);assert.match(sinkingBatch,/反常配合/);assert.match(sinkingBatch,/荒谬私利/);assert.match(sinkingBatch,/同等级、立即生效/);
  const normalBatch=buildOutlineDramaticBatchPrompt({total_episodes:10,tags:[],story_mode:"normal"},[],[],[],[{episode_no:1}],[],[]);assert.doesNotMatch(normalBatch,/反常配合/);
  const longBatch=buildOutlineDramaticBatchPrompt({total_episodes:10,tags:[],narrative_length:"long"},[],[],[],[{episode_no:1}],[],[]);assert.doesNotMatch(longBatch,/作品体量｜短篇/);
  const shortBatch=buildOutlineDramaticBatchPrompt({total_episodes:10,tags:[],narrative_length:"short"},[],[],[],[{episode_no:1}],[],[]);assert.match(shortBatch,/作品体量｜短篇/);assert.match(shortBatch,/单一处境/);assert.match(shortBatch,/1–2次.*宏观状态跃迁/);
  const shortNovel=buildEpisodeNovelPrompt({narrative_person:"first",emotion_intensity:"strong",narrative_length:"short"},[],[],{episode_no:1,summary:"主角被怪物堵门",required_plot:"怪物钻门→主角阻拦→门外出现熟人",must_not_reveal:"无"},[],{novelTarget:{}},{});assert.match(shortNovel,/单一处境内部充分设计/);
  const shortScript=buildSkillEpisodePrompt({narrative_person:"first",emotion_intensity:"strong",narrative_length:"short"},[],[],{episode_no:1,required_plot:"怪物钻门→主角阻拦",must_not_reveal:"无",novel:"我举刀阻拦怪物。"},[],{scriptTarget:{}},{episodePlan:"1 内 房间 夜\n（怪物钻门+主角举刀阻拦）"});assert.match(shortScript,/不得把短篇模式重新压缩成多项大事件/);
  assert.equal(novelPerspectiveIssue(`我推开门。\n\n「他怎么还没来？」`,"first"),"");
  assert.match(novelPerspectiveIssue(`方野推开门。\n\n「我来晚了。」`,"first"),/第一人称/);
  assert.equal(novelPerspectiveIssue(`方野推开门。\n\n「我来晚了。」`,"third"),"");
  assert.match(novelPerspectiveIssue(`我推开门。\n\n「你找谁？」`,"third"),/第三人称/);
  const safelySplit=splitNovelLongParagraphs("我推开院门，母亲从里屋走出来。她把缴费单递给我。门外突然响起急促的敲门声。刘麻子带着两个人堵住出口。");
  assert.equal(safelySplit,"我推开院门，母亲从里屋走出来。她把缴费单递给我。门外突然响起急促的敲门声。刘麻子带着两个人堵住出口。");assert.equal(novelDescriptionIssues(safelySplit).length,0);
  const indivisible="「你不知道还敢收我两千块的卦金，还敢保证这趟出行有惊无险，现在出了这么大的事你准备怎么负责？」";
  assert.equal(splitNovelLongParagraphs(indivisible),indivisible);assert.equal(novelDescriptionIssues(indivisible).length,0);
  assert.equal(novelDescriptionIssues("齐夏终于站起身，从乔家劲手中拿过羊皮面具，仔细看了看上面最后一句话。").length,0);
  assert.equal(novelDescriptionIssues(`我把面具拿到眼前，反复观察上面每一道已经干裂的纹路，又沿着边缘慢慢摸索，试图从那些细小痕迹里找出隐藏的信息和制造者留下的线索。`).length,0);
  assert.equal(novelDescriptionIssues(`王志远端起茶杯：「这只算见面礼，你敢不敢跟我去市场走一趟，把剩下那批原石全看完，再替我把所有鉴定结果逐件写清楚？」`).length,0);
  assert.equal(novelDescriptionIssues(`王志远端起茶杯：「这只算见面礼，敢不敢跟我去市场？」`).length,0);
  assert.equal(stripNovelRevisionHeading("# 修订后正文\n\n我推开门。"),"我推开门。");
  assert.equal((await request("/api/health")).body.ok, true);
  const iframeChild=await fetch(`${base}/vendor/iframeResizer.contentWindow.min.js`);assert.equal(iframeChild.status,200);assert.match(await iframeChild.text(),/iFrameResizer/);
  const home=await fetch(`${base}/`);assert.match(await home.text(),/iframeResizer\.contentWindow\.min\.js/);
  const connection=await request("/api/settings/llm/test",{method:"POST",body:JSON.stringify({provider:"mock"})});assert.equal(connection.body.ok,true);
  const providerSettings=(await request("/api/settings/llm")).body,baiduText=providerSettings.providers.find(item=>item.id==="baidu"),senseNovaText=providerSettings.providers.find(item=>item.id==="sensenova"),tencentText=providerSettings.providers.find(item=>item.id==="tencent_tokenhub"),tencentPlanText=providerSettings.providers.find(item=>item.id==="tencent_tokenplan"),aibhText=providerSettings.providers.find(item=>item.id==="aibh"),pinaiText=providerSettings.providers.find(item=>item.id==="pinai"),geeknowText=providerSettings.providers.find(item=>item.id==="geeknow");assert.equal(baiduText?.baseUrl,"https://qianfan.baidubce.com/v2");assert.equal(baiduText?.model,"ernie-5.0");assert.equal(senseNovaText?.baseUrl,"https://token.sensenova.cn/v1");assert.equal(senseNovaText?.model,"deepseek-v4-flash");assert.deepEqual(senseNovaText?.models?.map(item=>item.id),["deepseek-v4-flash","deepseek-v4-pro","glm-5.2","kimi-k3"]);assert.equal(tencentText?.baseUrl,"https://tokenhub.tencentmaas.com/v1");assert.equal(tencentText?.model,"glm-5.3");assert.equal(tencentPlanText?.baseUrl,"https://tokenhub.tencentmaas.com/plan/v3");assert.equal(aibhText?.baseUrl,"https://api.aibh.site/v1");assert.equal(pinaiText?.baseUrl,"https://api.pinai.cn/v1");assert.equal(geeknowText?.baseUrl,"https://geeknow.ai/v1");
  const embeddingConnection=await request("/api/settings/embedding/test",{method:"POST",body:JSON.stringify({provider:"mock"})});assert.equal(embeddingConnection.body.ok,true);assert.ok(embeddingConnection.body.dimensions>0);
  const modelSettings=(await request("/api/settings/llm")).body,baiduEmbedding=modelSettings.embeddingProviders.find(item=>item.id==="baidu");assert.equal(baiduEmbedding?.baseUrl,"https://qianfan.baidubce.com/v2");assert.equal(baiduEmbedding?.model,"bge-large-zh");
  const created=await request("/api/projects",{method:"POST",body:JSON.stringify({title:"测试短剧",total_episodes:6,tags:["悬疑"],seed:"失忆的女主每天收到未来的短信"})});
  assert.equal(created.r.status,201); const id=created.body.id;
  assert.equal((await request(`/api/projects/${id}`)).body.template_id,"newcomer-recommended");
  assert.equal((await request(`/api/projects/${id}`)).body.narrative_length,"long");
  const shortLength=await request(`/api/projects/${id}/narrative-length`,{method:"PUT",body:JSON.stringify({narrative_length:"short"})});assert.equal(shortLength.body.narrative_length,"short");assert.equal((await request(`/api/projects/${id}`)).body.narrative_length,"short");
  await request(`/api/projects/${id}/narrative-length`,{method:"PUT",body:JSON.stringify({narrative_length:"long"})});
  assert.deepEqual((await request(`/api/projects/${id}`)).body.idea_libraries,[]);
  assert.equal((await request(`/api/projects/${id}`)).body.idea_mode,"daily");
  const brainstormMode=await request(`/api/projects/${id}/idea-mode`,{method:"PUT",body:JSON.stringify({idea_mode:"brainstorm"})});assert.equal(brainstormMode.body.idea_mode,"brainstorm");assert.equal((await request(`/api/projects/${id}`)).body.idea_mode,"brainstorm");
  const brainstormPrompt=buildStagePrompt("idea",{...created.body,idea_mode:"brainstorm",tags:[]},[],[]);assert.match(brainstormPrompt,/冷门、小众身份/);assert.match(brainstormPrompt,/强反差/);assert.match(brainstormPrompt,/反差组合：A × B/);assert.match(brainstormPrompt,/反差如何持续产生剧情/);assert.match(brainstormPrompt,/core_hook必须实际发动这组反差/);assert.match(brainstormPrompt,/approved_idea必须将其融入最终方案/);assert.match(brainstormPrompt,/字段：diagnosis/);
  const sinkingPrompt=buildStagePrompt("idea",{...created.body,idea_mode:"sinking",tags:[]},[],[]);assert.match(sinkingPrompt,/爆点母句/);assert.match(sinkingPrompt,/凝练进 core_hook/);
  const laterPrompt=buildStagePrompt("planning",{...created.body,idea_mode:"sinking",tags:[]},[],[]);assert.doesNotMatch(laterPrompt,/爆点母句/);
  assert.match(laterPrompt,/01创意承接规则/);assert.match(laterPrompt,/core_hook、protagonist_goal、sustained_conflict/);assert.match(laterPrompt,/反差组合/);assert.match(laterPrompt,/不要原样抄写/);
  const characterPrompt=buildStagePrompt("characters",{...created.body,idea_mode:"brainstorm",tags:[]},[],[]);assert.doesNotMatch(characterPrompt,/01创意承接规则/);
  await request(`/api/projects/${id}/idea-mode`,{method:"PUT",body:JSON.stringify({idea_mode:"daily"})});assert.equal((await request(`/api/projects/${id}`)).body.idea_mode,"daily");
  const selectedIdeaLibraries=await request(`/api/projects/${id}/idea-libraries`,{method:"PUT",body:JSON.stringify({libraries:["reality","market","invalid","reality"]})});assert.deepEqual(selectedIdeaLibraries.body.libraries,["reality","market"]);assert.deepEqual((await request(`/api/projects/${id}`)).body.idea_libraries,["reality","market"]);
  const today=new Date().toISOString().slice(0,10),future=new Date(Date.now()+86400000).toISOString();run(`INSERT INTO knowledge_items(library,external_id,title,summary,snapshot_date,source_url,narrative_json,item_type,confidence,expires_at,auto_generated) VALUES('market','test-work','摆摊鉴宝逆袭','底层摊主获得鉴别古董真伪的能力',@date,'test://work',@narrative,'work_card',0.9,@expires,1)`,{date:today,expires:future,narrative:JSON.stringify({one_sentence_premise:"底层摊主依靠鉴宝能力进入古玩圈",core_mechanism:"鉴别古董真伪"})});const ragEvidence=await searchIdeaKnowledge({seed:"摆摊鉴宝",tags:["逆袭"],idea_libraries:["market"]},8);assert.ok(ragEvidence.some(item=>item.title==="摆摊鉴宝逆袭"));
  run(`INSERT INTO knowledge_items(library,external_id,title,summary,snapshot_date,source_url,narrative_json,item_type,confidence,expires_at,auto_generated) VALUES('reality','test-signal','特殊测试热搜标题','只是原始热搜证据',@date,'test://signal','{}','reality_signal',0.9,@expires,1),('reality','test-reality-trend','特殊测试现实趋势','多日多标题聚合得出的现实矛盾趋势',@date,'test://reality-trend',@trend,'reality_trend',0.9,@expires,1)`,{date:today,expires:future,trend:JSON.stringify({summary:"现实趋势摘要",transferable_mechanism:"冲突机制"})});const realityEvidence=await searchIdeaKnowledge({seed:"特殊测试",tags:["现实趋势"],idea_libraries:["reality"]},8);assert.ok(realityEvidence.some(item=>item.item_type==="reality_trend"));assert.ok(!realityEvidence.some(item=>item.item_type==="reality_signal"));
  run(`INSERT INTO knowledge_items(library,title,snapshot_date,source_url,item_type,expires_at,auto_generated) VALUES('market','过期资料',@date,'test://expired','work_card','2000-01-01T00:00:00.000Z',1)`,{date:today});assert.ok(cleanupAutomaticKnowledge()>=1);seedAutomaticSources();const knowledgeStatus=await request("/api/knowledge/status");assert.equal(knowledgeStatus.body.retentionDays,30);assert.ok(knowledgeStatus.body.sources.some(item=>item.name==="番茄小说官方榜单"));assert.ok(knowledgeStatus.body.sources.some(item=>item.name==="红果短剧官方榜单"&&item.enabled===0));
  await request(`/api/projects/${id}/idea-libraries`,{method:"PUT",body:JSON.stringify({libraries:[]})});assert.deepEqual((await request(`/api/projects/${id}`)).body.idea_libraries,[]);
  const builtinTemplate=(await request(`/api/projects/${id}`)).body.templates.find(item=>item.id==="default");
  const newcomerTemplate=(await request(`/api/projects/${id}`)).body.templates.find(item=>item.id==="newcomer-recommended");
  const zhihuTemplate=(await request(`/api/projects/${id}`)).body.templates.find(item=>item.id==="zhihu-yanxuan-manju");
  assert.deepEqual(builtinTemplate?.analysis?.inferredScriptFormat?.targetCharacters,{min:700,ideal:1050,max:1400});
  assert.deepEqual(builtinTemplate?.analysis?.inferredScriptFormat?.scriptAcceptance,{min:700,max:1400});
  assert.deepEqual(builtinTemplate?.analysis?.inferredScriptFormat?.targetScenes,{min:1,ideal:2,max:4});
  assert.equal(newcomerTemplate?.source,"builtin");
  assert.deepEqual(newcomerTemplate?.analysis?.inferredScriptFormat?.novelCharacters,builtinTemplate?.analysis?.inferredScriptFormat?.novelCharacters);
  assert.deepEqual(newcomerTemplate?.analysis?.inferredScriptFormat?.targetCharacters,builtinTemplate?.analysis?.inferredScriptFormat?.targetCharacters);
  assert.deepEqual(newcomerTemplate?.analysis?.inferredScriptFormat?.novelAcceptance,{min:0,max:5000});
  assert.deepEqual(newcomerTemplate?.analysis?.inferredScriptFormat?.scriptAcceptance,{min:0,max:5000});
  assert.equal(zhihuTemplate?.source,"builtin");
  assert.deepEqual(zhihuTemplate?.analysis?.inferredScriptFormat?.novelCharacters,{min:700,ideal:1000,max:1300});
  assert.deepEqual(zhihuTemplate?.analysis?.inferredScriptFormat?.targetCharacters,{min:500,ideal:700,max:900});
  assert.deepEqual(zhihuTemplate?.analysis?.inferredScriptFormat?.targetScenes,{min:1,ideal:2,max:4});
  assert.equal((await request(`/api/projects/${id}/template`,{method:"PUT",body:JSON.stringify({template_id:"zhihu-yanxuan-manju"})})).r.status,200);
  assert.equal((await request(`/api/projects/${id}`)).body.template_id,"zhihu-yanxuan-manju");
  assert.equal((await request(`/api/templates/zhihu-yanxuan-manju`,{method:"DELETE"})).r.status,400);
  await request(`/api/projects/${id}/template`,{method:"PUT",body:JSON.stringify({template_id:"default"})});
  const customFormat=structuredClone(builtinTemplate.analysis);customFormat.inferredScriptFormat.novelCharacters={min:1801,ideal:1999,max:2800};customFormat.inferredScriptFormat.novelAcceptance={min:1600,max:3200};customFormat.inferredScriptFormat.scriptAcceptance={min:1200,max:2200};customFormat.inferredScriptFormat.targetScenes={min:2,ideal:3,max:4};
  const createdFormatTemplate=await request("/api/templates",{method:"POST",body:JSON.stringify({source_mode:"default",project_id:id,name:"自定义篇幅格式",analysis:customFormat})});assert.equal(createdFormatTemplate.r.status,201);assert.equal(createdFormatTemplate.body.analysis.inferredScriptFormat.novelCharacters.ideal,2301);assert.equal(createdFormatTemplate.body.analysis.inferredScriptFormat.novelAcceptance.min,1600);assert.equal(createdFormatTemplate.body.analysis.inferredScriptFormat.novelAcceptance.max,3200);assert.equal(createdFormatTemplate.body.analysis.inferredScriptFormat.scriptAcceptance.min,1200);assert.equal(createdFormatTemplate.body.analysis.inferredScriptFormat.targetScenes.ideal,3);assert.equal(createdFormatTemplate.body.analysis.inferredScriptFormat.targetScenes.max,4);
  const afterFormatTemplate=(await request(`/api/projects/${id}`)).body;assert.ok(afterFormatTemplate.templates.some(item=>item.name==="自定义篇幅格式"));
  const scopedConstraint=await request(`/api/projects/${id}/constraints`,{method:"POST",body:JSON.stringify({category:"reveal",description:"只在第三至第五集生效",episode_start:3,episode_end:5})});assert.equal(scopedConstraint.r.status,201);assert.equal(scopedConstraint.body.episode_start,3);assert.equal(scopedConstraint.body.episode_end,5);
  const invalidConstraint=await request(`/api/projects/${id}/constraints`,{method:"POST",body:JSON.stringify({category:"reveal",description:"非法范围",episode_start:5,episode_end:3})});assert.equal(invalidConstraint.r.status,400);
  const thirdPerson=await request(`/api/projects/${id}/narrative-person`,{method:"PUT",body:JSON.stringify({narrative_person:"third"})});assert.equal(thirdPerson.body.narrative_person,"third");
  assert.equal((await request(`/api/projects/${id}`)).body.narrative_person,"third");
  const firstPerson=await request(`/api/projects/${id}/narrative-person`,{method:"PUT",body:JSON.stringify({narrative_person:"first"})});assert.equal(firstPerson.body.narrative_person,"first");
  await request(`/api/projects/${id}/constraints`,{method:"POST",body:JSON.stringify({category:"ending",description:"结局必须 HE"})});
  for(const stage of ["idea","planning","characters","cards","outline"]){
    const generated=await request(`/api/projects/${id}/generate/${stage}`,{method:"POST",body:"{}"}); assert.equal(generated.r.status,200); assert.equal(generated.body.generation.provider,"mock");
    const approved=await request(`/api/projects/${id}/artifacts/${stage}/approve`,{method:"POST",body:"{}"}); assert.equal(approved.r.status,200);
  }
  const project=(await request(`/api/projects/${id}`)).body; assert.equal(project.episodes.length,6); assert.equal(project.current_stage,"writing");
  const episode=await request(`/api/projects/${id}/episodes/1/generate`,{method:"POST",body:"{}"}); assert.doesNotMatch(episode.body.script,/^EP01/m);assert.match(episode.body.script,/^1\s+外/m);assert.ok(episode.body.novel);assert.match(episode.body.episode_plan,/【逻辑推理】/);assert.match(episode.body.character_identifiers_json,/主角/);
  const quality=await request(`/api/projects/${id}/episodes/1/quality`,{method:"POST",body:"{}"}); assert.equal(quality.body.modelReview.passed,true);assert.equal(typeof quality.body.metrics.characters,"number");
  const background=await request(`/api/projects/${id}/jobs/full-book`,{method:"POST",body:JSON.stringify({overwrite:false,auto_retry_limit:7})});assert.equal(background.r.status,202);assert.equal(background.body.auto_retry_limit,7);
  let job;for(let i=0;i<200;i++){await new Promise(r=>setTimeout(r,20));job=(await request(`/api/projects/${id}/jobs`)).body.find(x=>x.id===background.body.id);if(["completed","failed"].includes(job.status))break;}
  assert.equal(job.status,"completed",JSON.stringify({status:job.status,progress:job.progress,message:job.message,error:job.error,auto_retry_count:job.auto_retry_count,checkpoint:job.checkpoint}));assert.equal(job.progress,5);
  const fullBookHistory=(await request(`/api/projects/${id}/jobs/history`)).body.find(item=>item.id===background.body.id);assert.ok(fullBookHistory.recent_step_logs.some(item=>item.stage==="memory"&&item.outcome==="completed"),JSON.stringify(fullBookHistory.recent_step_logs));
  const completedProject=(await request(`/api/projects/${id}`)).body;assert.equal(completedProject.episodes.filter(x=>x.script).length,6);
  assert.equal(Number(get("SELECT COUNT(*) count FROM memory_event_scan_checkpoints WHERE project_id=@id",{id})?.count||0),0);
  const memoryPreview=await request(`/api/projects/${id}/memory-rebuild-preview`);assert.equal(memoryPreview.body.cutoff,6);assert.equal(memoryPreview.body.gap,null);
  const episode3=completedProject.episodes.find(item=>item.episode_no===3);await request(`/api/projects/${id}/episodes/3`,{method:"PUT",body:JSON.stringify({script:""})});const gapPreview=await request(`/api/projects/${id}/memory-rebuild-preview`);assert.equal(gapPreview.body.cutoff,2);assert.equal(gapPreview.body.gap,3);assert.equal(gapPreview.body.ignoredWritten,3);await request(`/api/projects/${id}/episodes/3`,{method:"PUT",body:JSON.stringify({script:episode3.script})});
  assert.equal(completedProject.storyMemory.stats.status,"ready");assert.ok(completedProject.storyMemory.stats.entities>=1);assert.equal(completedProject.storyMemory.stats.syncedEpisode,6);
  const exported=await request(`/api/projects/${id}/export`,{method:"POST",body:"{}"}); assert.match(exported.body.filename,/\.docx$/);
  const selectedScript=await request(`/api/projects/${id}/export`,{method:"POST",body:JSON.stringify({type:"script",episode_numbers:[1]})});assert.deepEqual(selectedScript.body.episodes,[1]);
  const scriptText=(await mammoth.extractRawText({path:path.join(process.env.APP_DATA_DIR,"exports",selectedScript.body.filename)})).value;assert.equal((scriptText.match(/EP01/g)||[]).length,1);
  const selectedNovel=await request(`/api/projects/${id}/export`,{method:"POST",body:JSON.stringify({type:"novel",episode_numbers:[1,2]})});assert.deepEqual(selectedNovel.body.episodes,[1,2]);
  const novelText=(await mammoth.extractRawText({path:path.join(process.env.APP_DATA_DIR,"exports",selectedNovel.body.filename)})).value;assert.doesNotMatch(novelText,/EP0[12]/);assert.match(novelText,/\n1\n/);assert.match(novelText,/\n2\n/);
  const selectedOutline=await request(`/api/projects/${id}/export`,{method:"POST",body:JSON.stringify({type:"outline",episode_numbers:[1,2]})});assert.deepEqual(selectedOutline.body.episodes,[1,2]);
  const outlineText=(await mammoth.extractRawText({path:path.join(process.env.APP_DATA_DIR,"exports",selectedOutline.body.filename)})).value;assert.match(outlineText,/分集梗概/);assert.match(outlineText,/EP01/);assert.match(outlineText,/本集大概内容/);assert.match(outlineText,/集尾钩子 \/ 悬念/);assert.match(outlineText,/必须发生/);
  const refusedClear=await request(`/api/projects/${id}/story-memory`,{method:"DELETE",body:JSON.stringify({confirm:false})});assert.equal(refusedClear.r.status,400);
  const cleared=await request(`/api/projects/${id}/story-memory`,{method:"DELETE",body:JSON.stringify({confirm:true})});assert.equal(cleared.r.status,200);assert.equal(cleared.body.cleared,true);assert.ok(cleared.body.records>0);
  const afterClear=(await request(`/api/projects/${id}`)).body;assert.equal(afterClear.storyMemory.stats.status,"empty");assert.equal(afterClear.storyMemory.stats.events,0);assert.equal(afterClear.storyMemory.stats.entities,0);assert.equal(afterClear.episodes.filter(item=>item.script).length,6);assert.ok(afterClear.artifacts.some(item=>item.type==="characters"));
  const workspaceBackup=createWorkspaceBackup("test");assert.equal(validateWorkspaceBackup(workspaceBackup).format,"mochao-workspace-backup");assert.equal(workspaceBackup.tables.projects.length,1);assert.equal(workspaceBackup.tables.episodes.length,6);assert.equal("jobs" in workspaceBackup.tables,false);assert.equal("generations" in workspaceBackup.tables,false);assert.doesNotMatch(JSON.stringify(workspaceBackup),/API_KEY/);run("UPDATE projects SET title='临时改名' WHERE id=@id",{id});const restoredWorkspace=restoreWorkspaceBackup(workspaceBackup);assert.equal(restoredWorkspace.projects,1);assert.equal((await request(`/api/projects/${id}`)).body.title,afterClear.title);assert.equal(restoredWorkspace.needsVectorRebuild,true);
  const workbenchOff=await request("/api/workbench");assert.equal(workbenchOff.body.parallelEnabled,false);assert.equal(workbenchOff.body.maxTasks,10);assert.equal(workbenchOff.body.effectiveConcurrency,1);
  const workbenchOn=await request("/api/workbench",{method:"PUT",body:JSON.stringify({parallel_enabled:true,concurrency_mode:"manual",concurrency_limit:2})});assert.equal(workbenchOn.body.parallelEnabled,true);assert.equal(workbenchOn.body.effectiveConcurrency,2);assert.match(workbenchOn.body.sessionId,/^wb-/);
  const workbenchFive=await request("/api/workbench",{method:"PUT",body:JSON.stringify({concurrency_mode:"manual",concurrency_limit:5})});assert.equal(workbenchFive.body.concurrencyLimit,5);assert.equal(workbenchFive.body.effectiveConcurrency,5);
  const workbenchAuto=await request("/api/workbench",{method:"PUT",body:JSON.stringify({concurrency_mode:"auto"})});assert.equal(workbenchAuto.body.concurrencyMode,"auto");
  const workbenchAutoCap=await request("/api/workbench",{method:"PUT",body:JSON.stringify({concurrency_mode:"auto",concurrency_limit:4})});assert.equal(workbenchAutoCap.body.concurrencyLimit,4);assert.equal(workbenchAutoCap.body.effectiveConcurrency,4);
  const workbenchDisabled=await request("/api/workbench",{method:"PUT",body:JSON.stringify({parallel_enabled:false})});assert.equal(workbenchDisabled.body.parallelEnabled,false);assert.equal(workbenchDisabled.body.sessionId,"");assert.deepEqual(workbenchDisabled.body.jobs,[]);
});

test.after(()=>server.close());
