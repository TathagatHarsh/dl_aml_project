import fs from 'node:fs/promises';
import path from 'node:path';
import {Presentation, PresentationFile} from '@oai/artifact-tool';
const root=path.resolve('phase2');
const d=JSON.parse(await fs.readFile(path.join(root,'submission/presentation_data.json'),'utf8'));
const {applyPresentationChartFont}=await import('/Users/tathagatharsh/.codex/plugins/cache/openai-primary-runtime/presentations/26.1105.12137/skills/presentations/container_tools/artifact_tool_utils.mjs');
const p=Presentation.create({slideSize:{width:1280,height:720}});
const font='Arial';
function text(s,t,x,y,w,h,size=28,color='#233847',bold=false){
 const a=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 a.text=t; a.text.style={fontFamily:font,fontSize:size,color,bold}; return a;
}
function slide(title,lines,notes=''){
 const s=p.slides.add();s.background.fill='#F7F9FA';
 text(s,title,64,48,1152,100,42,'#102F40',true);
 lines.forEach((l,i)=>text(s,l,70,180+i*100,1130,84,29));
 text(s,String(p.slides.items.length).padStart(2,'0'),1160,659,60,28,17,'#667985');
 s.speakerNotes.text=notes;return s;
}
slide('Predictive maintenance for industrial engines',[
 'Project 7 · Phase Two',
 'Engineered degradation features, LSTM sequences and hybrid fusion',
 'NASA C-MAPSS FD001 · Remaining useful life in operating cycles'
],'Team: MS Thejas, Deepak Pandey, Tathagat Harsh, Aryan, Garv Verma. Actual Phase Two contributions must be confirmed by the team.');
slide('Research question',[
 'Can sensor sequences capture degradation information lost by rolling summaries?',
 'Baseline: engineered features → Random Forest regression',
 'DL: 30-cycle sensor history → LSTM → remaining useful life',
 'Hybrid: fuse learned sequence state with explicit degradation features'
],'The hybrid is a hypothesis, not a guaranteed improvement. Phase One used a 125-cycle cap, retained explicitly here.');
slide('Data and fair evaluation',[
 '100 run-to-failure engines → 80 training + 20 validation engines',
 '100 separate official test engines → one prediction per final observed cycle',
 'Fit sensor selection and scaling on training engines only',
 'Report capped and uncapped RMSE / MAE; group errors by actual RUL'
],'Source: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/ . Split seed 2026. Test IDs are a separate namespace. Existing Phase One model used all 100 training engines and is not a fair direct comparator.');
slide('Causal inputs',[
 'Training RUL = final failure cycle − current cycle; cap = 125',
 'Features: cycle, sensor level, rolling mean / variability / slope, health change',
 'Windows stay within engines; short histories repeat the first observation',
 'FD001 has one regime: normalize using training-only statistics'
],'Rolling windows are 5 and 10 cycles. Health change means deviation from the first observed sensor value. No future reading is used. Test endpoint labels are supplied separately, not derived from test maxima.');
slide('LSTM and hybrid architectures',[
 '30 × S sensors → LSTM with 32 hidden units → final hidden state',
 'Sequence head: Linear(32) → ReLU → Dropout(0.2) → Linear(1)',
 'Hybrid branch: engineered features → Linear(16) → ReLU → concatenate',
 'Sigmoid output × 125 cycles; joint mean-squared-error training'
],'Source: Hochreiter and Schmidhuber (1997), https://doi.org/10.1162/neco.1997.9.8.1735 . Cell state c_t = f_t*c_previous + i_t*g_t; h_t = o_t*tanh(c_t). Hybrid head input width is 48. The model only sees the finite window, not all previous cycles.');
slide('Optimization and ablations',[
 'AdamW · learning rate 0.001 · batch 256 · gradient clipping at 1',
 'Weight decay 0.0001 · up to 60 epochs · early-stopping patience 10',
 'Three neural seeds; restore best validation checkpoint for each run',
 'Ablations: 1-cycle input, no head dropout, and added feature branch'
],'No-dropout retains other regularizers. Adding the hybrid branch changes parameter count, so this is not a parameter-matched causal test. Validation uses one fixed pseudo-truncation per engine. Literature alternative: Bai et al. (2018), https://arxiv.org/abs/1803.01271 .');
const results=d.summary.filter(x=>x.target==='capped');
const s=slide('Capped test errors',[], 'Source: phase2/reports/test_metrics.csv. Bars show mean per-run RMSE, not an ensemble metric. Neural models have three seeds; Random Forest has one run. Standard deviations are in the notebook and CSV.');
const chart=s.charts.add('bar',{position:{left:80,top:170,width:1100,height:390},categories:results.map(x=>x.model.replace('LSTM30_no_dropout','LSTM30\nno dropout')),series:[{name:'RMSE (cycles)',values:results.map(x=>x.rmse),fill:'#197C91'}],hasLegend:false,dataLabels:{showValue:true,textStyle:{fontSize:20}},xAxis:{textStyle:{fontSize:18}},yAxis:{title:{text:'RMSE (cycles)',textStyle:{fontSize:20}},textStyle:{fontSize:18}}});
applyPresentationChartFont(chart,{fontFamily:font});
text(s,'Selected using validation: '+d.selected,80,595,1110,50,26);
slide('Hybrid evidence',[
 'Compare Hybrid30 and LSTM30 on the same test engines',
 'Seed-ensemble RMSE difference: hybrid minus LSTM',
 'Paired engine-bootstrap 95% interval: '+d.ci.map(x=>x.toFixed(2)).join(' to ')+' cycles',
 d.conclusion
],'Source: phase2/reports/manifest.json. 2,000 paired resamples of whole engines. Negative difference favors hybrid. This ensemble comparison is different from the mean per-seed RMSE chart. The interval conditions on one training split.');
slide('Error by degradation stage',[
 'Uncapped MAE by stage: late ≤30 / middle 31–90 / early >90 cycles',
 ...['RandomForest','LSTM30','Hybrid30'].map(name=>name+': '+['late_0_30','middle_31_90','early_over_90'].map(stage=>d.stages.find(x=>x.model===name && x.stage===stage).mae.toFixed(2)).join(' / ')+' cycles')
],'Source: phase2/reports/stage_metrics.csv. Review stage-specific results in the notebook. No claim of real-engine readiness or calibrated prediction uncertainty. Overlapping windows are correlated; longer-lived engines contribute more training rows.');
slide('Reproducible demonstration',[
 'Open the executed notebook: split, architecture, curves and error tables',
 'Inspect manifest: engine IDs, data hashes, seeds and dependency versions',
 'Load a checkpoint and predict from sensor histories without labels',
 'Explain the measured conclusion and each member’s actual contribution'
],'Commands and acceptance checks are in phase2/README.md. Viva preparation is in phase2/VIVA.md. IEEE-format source is in phase2/submission/report.tex.');
await fs.mkdir(path.join(root,'.build/slides'),{recursive:true});
for(let i=0;i<p.slides.items.length;i++){
 const blob=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(root,'.build/slides',`${i+1}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const candidate=path.join(root,'.build/candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidate);
const skill='/Users/tathagatharsh/.codex/plugins/cache/openai-primary-runtime/presentations/26.1105.12137/skills/presentations';
const {finalizePresentation}=await import(path.join(skill,'container_tools/artifact_tool_utils.mjs'));
await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath:path.join(root,'submission/Phase2_Presentation_v2.pptx'),pythonExecutable:'/Users/tathagatharsh/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit'],fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,materializeLiteralChartWorkbooks:true,receiptPath:path.join(root,'.build/deck-validation-v2.json')});
console.log('Presentation complete');
