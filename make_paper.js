const fs = require('fs');
const d = require('docx');
const {Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
       Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
       PageNumber, Footer, TabStopType, convertInchesToTwip} = d;

const R = JSON.parse(fs.readFileSync('outputs/results.json','utf8'));
const E = JSON.parse(fs.readFileSync('outputs/results_extra.json','utf8'));
const LOFO = E.table8_lofo || [];
const ROC = E.roc || {}, CM = E.confusion || {};
const lofoMeanSup = LOFO.reduce((a,r)=>a+r.supervised_recall,0)/Math.max(1,LOFO.length);
const lofoMeanCas = LOFO.reduce((a,r)=>a+r.cascade_recall,0)/Math.max(1,LOFO.length);
const fig = (file, w, h) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing:{before:140, after:60},
  children:[new d.ImageRun({type:"png", data: fs.readFileSync(file),
                            transformation:{width:w, height:h}})]});

const FONT = "Times New Roman";
const CW = 9360;                       // usable width in DXA (Letter, 1" margins)

const P = (t, o={}) => new Paragraph({
  alignment: o.align || AlignmentType.JUSTIFIED,
  spacing: {after: o.after ?? 120, line: o.line ?? 240},
  indent: o.indent,
  children: (Array.isArray(t)?t:[t]).map(x =>
    typeof x === 'string'
      ? new TextRun({text:x, font:FONT, size:o.size||20, italics:o.i, bold:o.b})
      : x)
});
const RUN = (t,o={}) => new TextRun({text:t, font:FONT, size:o.size||20,
  bold:o.b, italics:o.i, superScript:o.sup});

const H1 = (n,t) => new Paragraph({
  heading: HeadingLevel.HEADING_1, spacing:{before:260, after:130},
  children:[new TextRun({text:`${n}. ${t}`, font:FONT, size:22, bold:true, color:"000000"})]});
const H2 = (n,t) => new Paragraph({
  heading: HeadingLevel.HEADING_2, spacing:{before:180, after:100},
  children:[new TextRun({text:`${n}. ${t}`, font:FONT, size:20, bold:true, italics:true, color:"000000"})]});

const cap = (t) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing:{before:60, after:180},
  children:[new TextRun({text:t, font:FONT, size:17})]});

function table(headers, rows, widths, opts={}){
  const total = widths.reduce((a,b)=>a+b,0);
  const cols = widths.map(w=>Math.round(w/total*CW));
  const cell = (txt, o={}) => new TableCell({
    width:{size:cols[o.i], type:WidthType.DXA},
    shading:o.sh?{type:ShadingType.CLEAR, fill:o.sh, color:"auto"}:undefined,
    margins:{top:40,bottom:40,left:70,right:70},
    children:[new Paragraph({
      alignment:o.a||AlignmentType.LEFT, spacing:{after:0},
      children:[new TextRun({text:String(txt), font:FONT, size:16, bold:o.b})]})]});
  return new Table({
    columnWidths: cols,
    width:{size:CW, type:WidthType.DXA},
    rows:[
      new TableRow({tableHeader:true, children:headers.map((h,i)=>
        cell(h,{i, b:true, sh:"E7ECF0", a:i?AlignmentType.CENTER:AlignmentType.LEFT}))}),
      ...rows.map(r=>new TableRow({children:r.map((v,i)=>
        cell(v,{i, a:i?AlignmentType.RIGHT:AlignmentType.LEFT,
                sh:opts.hl && opts.hl(r)?"F1F5F8":undefined}))}))
    ]});
}

const f2 = v => Number(v).toFixed(2);
const t1 = R.table1_model_comparison;
const t2 = R.table2_evgf_ablation;
const t3 = R.table3_zero_day;
const t4 = R.table4_adversarial;
const t5 = R.table5_cost;
const t6 = R.table6_drift;
const lat = R.table7_latency;
const g3 = k => t3.find(r=>r.Algorithm===k);
const zdSup = g3("Supervised only - ZERO-DAY family");
const zdCas = g3("AdShield-X cascade - ZERO-DAY family");
const inSup = g3("Supervised only - in-distribution");
const inCas = g3("AdShield-X cascade - in-distribution");
const ablF = t2.find(r=>r.Algorithm.includes("full"));
const ablN = t2.find(r=>r.Algorithm.includes("behaviour + network"));
const ablB = t2.find(r=>r.Algorithm.includes("behaviour only"));
const advLast = t4[t4.length-1];

const body = [];
const add = (...x)=>body.push(...x);

/* ------------------------------- title ---------------------------------- */
add(new Paragraph({alignment:AlignmentType.CENTER, spacing:{after:120},
  children:[new TextRun({text:"AdShield-X: Entity-Graph Features, Zero-Day Novelty Detection and Revenue-Aware Thresholding for Ad Click Fraud", font:FONT, size:32, bold:true})]}));
add(new Paragraph({alignment:AlignmentType.CENTER, spacing:{after:60},
  children:[RUN("Author Name", {size:22})]}));
add(new Paragraph({alignment:AlignmentType.CENTER, spacing:{after:280},
  children:[RUN("Department of Computer Science and Engineering", {size:18, i:true}),
            RUN("\u2003·\u2003Institution\u2003·\u2003email@institution.edu", {size:18, i:true})]}));

/* ------------------------------ abstract -------------------------------- */
add(P([RUN("Abstract—", {b:true}),
 RUN(`Pay-per-click advertising loses a large share of its spend to automated clicks, and the published machine-learning literature on the problem has converged on a single experimental shape: encode a click record, train ten to fourteen classifiers, and report accuracy on a random split. That protocol answers a question advertisers do not have. It measures how well a model recognises fraud that already appeared in its training labels, on a random split, against an adversary assumed to be static, at a decision threshold chosen without reference to what a click costs. This paper reproduces that baseline and then attacks the four assumptions underneath it. We contribute (i) EVGF, a set of leakage-free entity-velocity graph features computed over a backward-only window across the IP–device–publisher graph; (ii) a two-head cascade pairing a supervised detector with a denoising autoencoder trained exclusively on legitimate clicks, gated to a fixed false-positive budget; (iii) an exemplar-replay mimicry attack that perturbs only signals a fraud operator actually controls, together with adversarial training as a defence; (iv) a cost-optimal decision threshold derived from cost-per-click and the price of blocking a genuine customer; and (v) a drift-monitored deployment loop. On a `),
 RUN(`${R.dataset.rows.toLocaleString()}`,{}), RUN(`-click benchmark with four bot families and 2% label noise, EVGF lifts F1 from ${f2(ablB["F1-Score"])} to ${f2(ablF["F1-Score"])} while cutting the false-positive rate from ${f2(ablB.FPR)}% to ${f2(ablF.FPR)}%. Against a bot family withheld from training, the supervised model recalls ${f2(zdSup.Recall)}% and the cascade recalls ${f2(zdCas.Recall)}% at ${f2(zdCas.Precision)}% precision; under a leave-one-family-out protocol across all four families, mean recall on the unknown family rises from ${f2(lofoMeanSup)}% to ${f2(lofoMeanCas)}%, concentrated entirely in the family that is structurally unlike the training set. Under full-fidelity mimicry, a behaviour-only detector's attack-success rate reaches ${f2(advLast["ASR_behaviour_only_%"])}%; with EVGF it is held to ${f2(advLast["ASR_plus_EVGF_%"])}% and with adversarial training to ${f2(advLast["ASR_plus_EVGF_advtrain_%"])}%. Revenue-aware thresholding recovers ₹${Number(t5[0]["saving_per_million_clicks_INR"]).toLocaleString('en-IN')} per million clicks over the conventional 0.50 cut. We argue these axes, not another decimal place of accuracy, are where the remaining headroom in click-fraud detection lies.`,{})]));
add(P([RUN("Index Terms—",{b:true}), RUN("click fraud, online advertising, graph features, novelty detection, adversarial machine learning, concept drift, cost-sensitive learning.",{i:true})], {after:200}));

/* ------------------------------- I ------------------------------------- */
add(H1("I","Introduction"));
add(P(`Pay-per-click is the settlement layer of the display advertising economy. A publisher hosts a creative, a visitor clicks it, and the advertiser is billed. The arrangement only works while the click is evidence that a person saw the advertisement, and that assumption is what invalid traffic attacks. An operator who can manufacture clicks converts an advertiser's budget into publisher revenue directly, and the advertiser pays not only the click price but the downstream cost of an audience model trained on phantom interest.`));
add(P(`Early countermeasures were rules: thresholds on inter-click intervals, blocklists of offending addresses, caps on clicks per session. Rules are cheap to run and cheap to evade, because a rule is a published specification of what the defender will not tolerate. The field moved to supervised learning, and a now-standard experimental template emerged. Alzahrani, Aljabri and Mohammad [1] give the clearest recent example: ten classical learners and three deep architectures are trained on a labelled click corpus, accuracy, precision, recall and F1 are tabulated, an attention-augmented convolutional network is proposed as an extension, and accuracies above 98% are reported. Several 2025 and 2026 papers repeat this design almost exactly [2], [3], [4].`));
add(P(`We do not dispute those numbers. We question what they measure. Four assumptions are built into the protocol and none of them survive contact with a live advertising exchange:`));
add(P([RUN("Fraud resembles its training labels. ",{i:true}), RUN("A random split places clicks from the same bot campaign on both sides of the partition, so held-out performance largely reports memorisation of families the model has already met. A new campaign is, definitionally, out of distribution.")], {indent:{left:convertInchesToTwip(0.25)}}));
add(P([RUN("A click can be judged alone. ",{i:true}), RUN("Click fraud is coordinated: a farm reuses a limited pool of addresses and devices and concentrates on a few publishers. Row-wise features cannot see coordination, so the strongest available evidence is discarded before learning begins.")], {indent:{left:convertInchesToTwip(0.25)}}));
add(P([RUN("The adversary is static. ",{i:true}), RUN("The features these models rely on most heavily are browser-side telemetry, and browser-side telemetry is precisely what an operator can forge. A detector's reported accuracy is an upper bound taken against an opponent who is not trying.")], {indent:{left:convertInchesToTwip(0.25)}}));
add(P([RUN("Accuracy is the objective. ",{i:true}), RUN("It is not. Blocking a genuine customer costs the click price plus the forgone conversion; paying for one invalid click costs the click price. The two errors are not interchangeable, so the implicit 0.50 threshold is not the operating point any advertiser wants.")], {indent:{left:convertInchesToTwip(0.25)}}));
add(P(`This paper takes each assumption in turn. Section III describes AdShield-X, a detection pipeline that adds an entity-graph feature layer, a novelty head trained only on legitimate behaviour, an adversarial evaluation protocol, a cost-derived threshold and a drift-triggered retraining loop. Section IV reproduces the standard fourteen-model comparison so our results sit on the same scale as prior work, then reports five experiments that the standard protocol cannot express. Section V discusses what the results imply for deployment, including two findings that run against the literature.`));

/* ------------------------------- II ------------------------------------ */
add(H1("II","Related Work"));
add(H2("A","Click fraud detection with supervised learning"));
add(P(`Aljabri and Mohammad [5] separate human from automated browsing using session-level features and report Random Forest as the strongest of several classifiers. Batool and Byun [6] ensemble deep architectures for pay-per-click campaigns. Thejas et al. [7], [8] combine supervised and unsupervised components and study multi-time-scale behaviour on the BuzzCity corpus. Alzahrani et al. [1] give the broadest recent comparison and add attention to a convolutional model. Across this line of work tree ensembles and gradient boosting are consistently at or near the top, and reported accuracies cluster between 97% and 99%.`));
add(P(`Two limitations recur. First, evaluation is on random splits of a single labelled corpus, so nothing is said about families absent from that corpus. Second, the feature sets are dominated by client-side session telemetry. The reviews of Sisodia and Sisodia [9] and the AI-based survey in [10] both note that adversarial adaptation and evolving fraud patterns remain open, but neither line of work is evaluated in the click-fraud setting.`));
add(H2("B","Robustness, drift and cost in adjacent fraud domains"));
add(P(`The financial-fraud literature has taken these questions further. Hybrid frameworks combining cost-sensitive learning, adversarial training and drift detectors such as DDM and ADWIN have been reported for transaction monitoring, with explicit robustness and drift-recovery metrics [11]. Graph neural approaches with reinforcement-learned thresholds have been proposed for the same domain [12]. Surveys of adversarial learning in fraud detection [13] observe that most adversarial-robustness research addresses vision and language rather than the tabular data on which fraud systems actually run, and call for synthetic generators that allow attacks to be tested against drift and delayed feedback.`));
add(P(`To our knowledge these threads have not been brought together for advertising click fraud, and no click-fraud paper we are aware of reports attack-success rate under a constrained adversary, recall against a withheld bot family, or a threshold derived from cost-per-click. That gap is the subject of this paper.`));

/* ------------------------------- III ----------------------------------- */
add(H1("III","Proposed Method"));
add(P(`AdShield-X is organised as five components over a common feature pipeline, shown in Fig. 1. The white blocks are the pipeline that prior work already has; the numbered blocks are what this paper adds. Two dashed paths carry most of the argument: the red path marks what a fraud operator can reach, and the green path is the loop that keeps the deployed model current.`));
add(fig('outputs/architecture.png', 620, 380));
add(cap("Fig. 1.  The AdShield-X pipeline. Numbered blocks are the contributions of this paper."));

add(H2("A","Entity-velocity graph features (EVGF)"));
add(P(`Each click carries three entity identifiers: a hashed source address, a hashed device fingerprint and a publisher identifier. Together these induce a tripartite graph. Rather than learn on that graph directly, which would require the whole graph at inference time, we attach to each click a summary of its own local neighbourhood as it stood immediately before the click occurred.`));
add(P(`Concretely, for each click at time t we maintain per-entity event deques and compute ten quantities over a short (1 h) and a long (24 h) backward window: click velocity for the address and for the device; the number of distinct publishers the address has touched; the number of distinct addresses the device has used; publisher click velocity; the ratio of distinct addresses to total clicks at that publisher; the Herfindahl concentration of that publisher's traffic across addresses; a burstiness term equal to the reciprocal of the median inter-arrival gap on that address; a repeat count for the exact address–device–publisher triple; and a log-scaled joint degree term.`));
add(P(`Two properties matter. All windows look strictly backwards, so no future information enters a training row and the features are leakage-free by construction. And every quantity is maintained by O(1) streaming counters, so the layer is deployable rather than merely computable offline: our single-threaded reference implementation sustains ${R.evgf_build_rate_clicks_per_s.toLocaleString()} clicks per second.`));
add(P(`The distinction that carries the adversarial results is that EVGF features are server-side. A fraud operator controls its browser and can forge any client telemetry it likes, but it cannot edit the exchange's own counters, and it cannot lower its click velocity without lowering its revenue.`));

add(H2("B","Zero-day cascade"));
add(P(`A supervised model estimates P(fraud | x) from labelled examples and therefore recognises what it has been shown. We add a second head that models legitimate behaviour instead. A denoising autoencoder with a 64–20–64 bottleneck is trained on legitimate clicks only, after a rank-normalising quantile transform; the transform matters because the raw behavioural features are heavy-tailed and an untransformed squared error is dominated by their outliers rather than by the structure we want.`));
add(P(`The autoencoder learns how signals co-vary in a real session: that a desktop client emits no touch events, that a visitor who spends a minute across seven pages has also scrolled, that a browser advertising a rich user-agent also accepts cookies. A bot family that copies each marginal distribution but not the couplings between them reconstructs badly.`));
add(P(`A click is escalated to fraud when the supervised head is not already confident it is fraud (p < 0.55) and its reconstruction error exceeds a gate. The gate is set at the (1 − β) quantile of reconstruction error on known-legitimate traffic, with β = ${R.cascade_config.human_fp_budget} in our experiments. This makes the price of the second head a parameter the operator chooses in advance rather than a property they discover in production.`));
add(P(`The novelty head deliberately receives the unreduced feature matrix. Feature selection is optimised against known fraud and therefore discards exactly the columns on which an unseen family betrays itself; we quantify this below.`));

add(H2("C","Exemplar-replay mimicry attack"));
add(P(`To evaluate against an adaptive adversary we need an attack a real operator could mount. Perturbing a row toward the mean of the legitimate population is not one: averaging heavy-tailed behavioural features produces a profile no visitor has. What an operator actually does is capture genuine sessions and replay their profiles. We model this directly. For each fraudulent click we draw a random legitimate exemplar and interpolate:`));
add(P([RUN("x′ = (1 − b)·x", {i:true}), RUN("bot",{i:true,sup:false}), RUN(" + b·x",{i:true}), RUN("exemplar",{i:true}), RUN(" + ε,\u2003ε ~ 0.05·σ",{i:true}), RUN("legit",{i:true}), RUN("·N(0,1)",{i:true})], {align:AlignmentType.CENTER, after:140}));
add(P(`The interpolation applies only to controllable columns: session and dwell timings, pointer and touch counts, scroll depth, keystrokes, click cadence and its dispersion, user-agent entropy, client capability flags and the categorical browser, operating-system, device and referrer fields. Velocity, publisher-concentration and post-click economic features are held fixed, because an attack permitted to rewrite the defender's own counters is not an attack but a database compromise. The budget b ∈ [0,1] expresses replay fidelity, which costs the operator engineering effort and throughput and is therefore an honest axis of adversary sophistication. We report attack-success rate, the share of true bots scored below the decision threshold. As a defence we augment training with attacked copies of the fraudulent rows at b ∈ {0.3, 0.6, 0.9}.`));

add(H2("D","Revenue-aware decision threshold"));
add(P(`Let c_FN be the cost of paying for one invalid click, equal to the cost-per-click, and c_FP the cost of blocking a genuine customer, equal to the click price plus the forgone margin. Expected loss per click at threshold τ is L(τ) = [FP(τ)·c_FP + FN(τ)·c_FN] / N. We sweep τ over [0.01, 0.99] and select the minimiser. Because c_FP > c_FN, the optimum sits above 0.50, and the gap between L(0.50) and L(τ*) is budget that accuracy-optimised systems leave on the table.`));

add(H2("E","Drift monitoring and retraining"));
add(P(`Deployed detectors face a non-stationary opponent. We monitor the streaming error signal with a two-window change detector: a reference window and a recent window, with drift declared when the difference in mean error exceeds a Hoeffding-style bound at confidence δ. On a drift event the model is refitted on the original training set plus the most recent 4,000 labelled stream observations. Retraining is triggered by observed change rather than by a schedule, so the system is idle while the adversary is.`));

add(H2("F","Reason codes"));
add(P(`An advertiser cannot dispute an invalid-traffic charge with a probability. We attach to each flagged click the top-k SHAP contributions, mapped to plain-language phrases and signed by direction, so that a decision can be defended: near-absent pointer activity, machine-regular click timing, a page never scrolled below the fold, a device rotating through many addresses.`));

/* ------------------------------- IV ------------------------------------ */
add(H1("IV","Experimental Setup"));
add(H2("A","Data"));
add(P(`The corpus used by the base paper [1] is licensed from a commercial trust network and is neither free nor published, which makes that work unreproducible as specified. We therefore evaluate on a behavioural simulator whose generative assumptions are stated explicitly and which the accompanying software ships in full. The pipeline is schema-agnostic and runs unchanged on a real labelled CSV.`));
add(P(`The simulator produces ${R.dataset.rows.toLocaleString()} clicks (${R.dataset.class_counts.Human.toLocaleString()} legitimate, ${R.dataset.class_counts.Bot.toLocaleString()} fraudulent) across ${R.dataset.raw_columns} raw columns, and is designed to be hard rather than flattering. Three properties do that work. First, there are four bot families rather than one: crude headless automation; jittered automation with randomised timings; mimicry bots that copy human behavioural marginals; and a residential-proxy click farm, held out of training entirely and used only for the zero-day experiment. Second, there is genuine class overlap: mobile and tablet visitors generate almost no pointer-move events and short sessions, which is the exact signature naive detectors use to flag bots and the dominant source of false positives in production. Third, labels carry 2% noise, reflecting that ground truth comes from delayed and imperfect conversion signals.`));
add(P(`After EVGF the matrix has ${R.all_features.length} features; recursive feature elimination with a random-forest estimator retains ${R.rfe_selected_features.length}. We use a stratified 80/20 split, and the kernel SVM is fitted on a 10,000-row subsample because its training cost is superlinear.`));

add(H2("B","Models and metrics"));
add(P(`We reproduce the fourteen-model comparison of [1]: logistic regression, decision tree, random forest, k-nearest neighbours, a multilayer perceptron, gradient boosting, LightGBM, XGBoost, Gaussian naive Bayes, an RBF SVM, and four deep models — a fully connected network, a 1-D convolutional network, a recurrent network and the attention-augmented convolutional network proposed as the extension in [1]. The deep models are implemented on a compact reverse-mode automatic-differentiation engine written for this project, gradient-checked against finite differences, so the entire study reproduces with NumPy, pandas and scikit-learn and no accelerator. Beyond accuracy, precision, recall and F1 we report ROC-AUC, PR-AUC, Matthews correlation and, because it is the metric an advertiser feels, the false-positive rate.`));

/* ------------------------------- V ------------------------------------- */
add(H1("V","Results"));

add(H2("A","Baseline comparison"));
add(P(`Table I reports all fourteen models on the held-out split with EVGF present for every model. XGBoost leads at ${f2(t1[0].Accuracy)}% accuracy and ${f2(t1[0]["F1-Score"])} F1, followed by LightGBM and random forest. The ordering matches the tree-dominance reported throughout the literature.`));
add(table(
  ["Algorithm","Acc.","Prec.","Rec.","F1","AUC","FPR","MCC"],
  t1.map(r=>[r.Algorithm, f2(r.Accuracy), f2(r.Precision), f2(r.Recall),
             f2(r["F1-Score"]), f2(r["ROC-AUC"]), f2(r.FPR)+"%", r.MCC.toFixed(3)]),
  [30,9,9,9,9,9,9,10]));
add(cap("TABLE I.  Fourteen-model comparison on the held-out split, sorted by F1."));
add(P(`Two observations depart from prior reports. The attention-augmented convolutional model, presented as the extension in [1], is the second-weakest model here at ${f2(t1.find(r=>r.Algorithm.startsWith("CNN+Attention"))["F1-Score"])} F1 and the worst on false positives at ${f2(t1.find(r=>r.Algorithm.startsWith("CNN+Attention")).FPR)}%. The reason is structural rather than a tuning failure: a 1-D convolution assumes that adjacent inputs are related, and the column order of a click record is arbitrary. Gains reported for this architecture are, we suspect, specific to a particular column ordering. Second, the spread across models is narrow — eleven of fourteen fall within three points of F1 — which is precisely why a comparison table is a weak instrument for choosing a detector, and why the experiments that follow separate systems the table cannot.`));

add(H2("B","Value of the entity graph"));
add(P(`Table II holds the classifier fixed and varies the feature set. Client-side behaviour alone reaches ${f2(ablB["F1-Score"])} F1 with a ${f2(ablB.FPR)}% false-positive rate. Adding network and economic signals lifts this to ${f2(ablN["F1-Score"])} F1 at ${f2(ablN.FPR)}%. The full set with EVGF reaches ${f2(ablF["F1-Score"])} F1 at ${f2(ablF.FPR)}%.`));
add(table(
  ["Feature set","Acc.","Prec.","Rec.","F1","FPR","Real visitors blocked"],
  [[ "Behavioural only", f2(ablB.Accuracy), f2(ablB.Precision), f2(ablB.Recall), f2(ablB["F1-Score"]), f2(ablB.FPR)+"%", ablB.FP],
   [ "+ network / economic", f2(ablN.Accuracy), f2(ablN.Precision), f2(ablN.Recall), f2(ablN["F1-Score"]), f2(ablN.FPR)+"%", ablN.FP],
   [ "+ EVGF (full)", f2(ablF.Accuracy), f2(ablF.Precision), f2(ablF.Recall), f2(ablF["F1-Score"]), f2(ablF.FPR)+"%", ablF.FP]],
  [26,10,10,10,10,10,17]));
add(cap("TABLE II.  Feature ablation with a random forest held constant."));
add(P(`The headline gain of ${f2(ablF["F1-Score"]-ablB["F1-Score"])} F1 understates the operational effect. False positives fall by ${((1-ablF.FPR/ablB.FPR)*100).toFixed(0)}%, from ${ablB.FP} wrongly blocked visitors to ${ablF.FP} on the same test set. A false positive is a real customer refused, and it is the error advertisers escalate.`));

add(H2("C","An unseen bot family"));
add(P(`We now test against the residential-proxy family withheld from training. Its marginal distributions are drawn from real human traffic and its per-address velocity is unremarkable by design; what identifies it is the joint structure of the template, including desktop sessions emitting touch events, minute-long visits that never scroll, and an invariant user-agent build.`));
add(table(
  ["Configuration","Recall","Precision","F1","FPR"],
  [["Supervised only, in-distribution", f2(inSup.Recall)+"%", f2(inSup.Precision)+"%", f2(inSup["F1-Score"]), f2(inSup.FPR)+"%"],
   ["Supervised only, zero-day family", f2(zdSup.Recall)+"%", f2(zdSup.Precision)+"%", f2(zdSup["F1-Score"]), f2(zdSup.FPR)+"%"],
   ["AdShield-X cascade, zero-day family", f2(zdCas.Recall)+"%", f2(zdCas.Precision)+"%", f2(zdCas["F1-Score"]), f2(zdCas.FPR)+"%"],
   ["AdShield-X cascade, in-distribution", f2(inCas.Recall)+"%", f2(inCas.Precision)+"%", f2(inCas["F1-Score"]), f2(inCas.FPR)+"%"]],
  [40,15,15,15,15]));
add(cap("TABLE III.  Performance against one bot family absent from training."));
add(P(`The supervised detector, which recalls ${f2(inSup.Recall)}% of in-distribution fraud, recalls ${f2(zdSup.Recall)}% of this family: ${zdSup.FN.toLocaleString()} of ${(zdSup.FN+zdSup.TP).toLocaleString()} fraudulent clicks are passed through and billed. This is the practical meaning of a random-split accuracy figure, and it is not visible anywhere in Table I. Adding the novelty head raises recall to ${f2(zdCas.Recall)}% at ${f2(zdCas.Precision)}% precision. The cost is ${f2(inSup["F1-Score"]-inCas["F1-Score"])} F1 on ordinary traffic, from ${f2(inSup["F1-Score"])} to ${f2(inCas["F1-Score"])}, which is the false-positive budget β behaving as configured.`));
add(P(`We also confirmed the motivation for giving the novelty head the unreduced matrix. An isolation forest on the RFE-selected features flagged under 2% of this family at the same budget; the autoencoder on the full matrix flags ${f2(zdCas.Recall)}%. Axis-aligned isolation cannot express the cross-feature inconsistencies that identify a template, and feature selection tuned to known fraud removes the columns in which they appear.`));

add(H2("D","Leave-one-family-out"));
add(P(`Holding out a single family answers a narrow question. We therefore repeat the protocol with each family withheld in turn: the detector is trained on legitimate clicks plus the other three families, feature selection is refitted inside every fold so the withheld family contributes at no stage, and recall is measured on the withheld family alone. Table IV reports all four folds.`));
add(table(
  ["Withheld family","Clicks","Supervised recall","Cascade recall","Gain","In-dist. F1"],
  LOFO.map(r=>[r.family_label, r.held_out_clicks.toLocaleString(),
               f2(r.supervised_recall)+"%", f2(r.cascade_recall)+"%",
               (r.cascade_recall-r.supervised_recall>0?"+":"")+f2(r.cascade_recall-r.supervised_recall),
               f2(r.in_dist_f1_supervised)+" → "+f2(r.in_dist_f1_cascade)]),
  [26,11,17,15,11,20]));
add(cap("TABLE IV.  Leave-one-family-out. Each family is withheld from training in turn."));
add(P(`Mean recall on the unknown family rises from ${f2(lofoMeanSup)}% to ${f2(lofoMeanCas)}%, a gain of ${f2(lofoMeanCas-lofoMeanSup)} points. The distribution of that gain is the more useful result. Where the withheld family is a variation on families the model has already seen — crude and jittered automation differ mainly in how much timing jitter they inject — the supervised model already recalls above 95% and the novelty head adds almost nothing. Where the withheld family is structurally different, the picture inverts: on the residential proxy farm the supervised model recalls ${f2(LOFO.find(r=>r.family==='proxy_farm').supervised_recall)}% and the cascade ${f2(LOFO.find(r=>r.family==='proxy_farm').cascade_recall)}%.`));
add(P(`We read this as a boundary on the claim rather than a weakness of it. A novelty head does not make a detector robust to unseen fraud in general; it makes it robust to unseen fraud that occupies a different region of behaviour space. The mimicry fold, where the supervised model falls to ${f2(LOFO.find(r=>r.family==='mimicry').supervised_recall)}% and the cascade reaches only ${f2(LOFO.find(r=>r.family==='mimicry').cascade_recall)}%, is the honest counter-example: a family engineered to sit inside the legitimate manifold is missed by a model of that manifold, and it is the entity-graph layer rather than the novelty head that catches it.`));

add(H2("E","Where the errors land"));
add(P(`Fig. 2 shows ROC curves for the strongest models, restricted to the corner an operator would actually deploy in; over the full unit square the curves are indistinguishable, which is itself a comment on the value of an AUC column. Table V gives the confusion matrices at the deployed threshold.`));
add(fig('outputs/roc.png', 400, 286));
add(cap("Fig. 2.  ROC detail in the low-false-positive region."));
add(table(
  ["Model","Threshold","Real visitors blocked","Bots billed","AUC"],
  Object.entries(CM).map(([k,c])=>[k, c.threshold, String(c.fp), String(c.fn),
                                   f2((ROC[k]||{}).auc||0)]),
  [30,14,22,18,16]));
add(cap("TABLE V.  Confusion outcomes on the held-out split at the deployed threshold."));
add(P(`The attention-augmented network is separated here in a way the accuracy column obscures. Its AUC of ${f2((ROC["CNN+Attention"]||{}).auc||0)} is within half a point of the best model, yet it blocks ${(CM["CNN+Attention"]||{}).fp} genuine visitors against ${(CM["XGBoost"]||{}).fp} for XGBoost — a factor of ${(( (CM["CNN+Attention"]||{}).fp||1)/((CM["XGBoost"]||{}).fp||1)).toFixed(1)}. Ranking quality and calibration at a usable operating point are different properties, and only the second one bills anybody.`));

add(H2("F","Adversarial robustness"));
add(P(`Table VI reports attack-success rate as replay fidelity increases, for three defences.`));
add(table(
  ["Replay fidelity b","Behavioural only","+ EVGF","+ EVGF and adversarial training"],
  t4.map(r=>[r.attack_budget.toFixed(1), f2(r["ASR_behaviour_only_%"])+"%",
             f2(r["ASR_plus_EVGF_%"])+"%", f2(r["ASR_plus_EVGF_advtrain_%"])+"%"]),
  [22,26,20,32]));
add(cap("TABLE VI.  Attack-success rate under exemplar-replay mimicry."));
add(P(`The behaviour-only detector, which is the standard formulation in the click-fraud literature, degrades from ${f2(t4[0]["ASR_behaviour_only_%"])}% to ${f2(advLast["ASR_behaviour_only_%"])}% attack-success rate: at full replay fidelity nearly nine bots in ten pass. This is the cost of building a detector entirely on signals the adversary owns. With EVGF the same attack reaches only ${f2(advLast["ASR_plus_EVGF_%"])}%, because the operator would have to suppress its own click velocity and publisher concentration to move those features, and doing so suppresses its revenue. Adversarial training reduces this further to ${f2(advLast["ASR_plus_EVGF_advtrain_%"])}% for ${f2(R.adversarial_clean_cost.clean_F1_plus_EVGF - R.adversarial_clean_cost.clean_F1_plus_EVGF_advtrain)} F1 on clean traffic, with the false-positive rate moving from ${f2(R.adversarial_clean_cost.clean_FPR_plus_EVGF)}% to ${f2(R.adversarial_clean_cost.clean_FPR_plus_EVGF_advtrain)}%.`));
add(P(`We read this as the central practical finding of the paper. Robustness here is not a property of the classifier; it is a property of where the features come from. Server-side observables are expensive for the adversary to move, and a detector built on them degrades gracefully where one built on browser telemetry collapses.`));

add(H2("G","Concept drift"));
add(P(`Table VII follows a stream of four blocks in which the bot mixture shifts toward evasive families, with the proxy farm appearing in the final block.`));
add(table(
  ["Block","Composition","Frozen model","Drift-monitored model"],
  [["0","mostly crude bots", f2(t6[0]["accuracy_static_%"])+"%", f2(t6[0]["accuracy_adaptive_%"])+"%"],
   ["1","shifting mixture", f2(t6[1]["accuracy_static_%"])+"%", f2(t6[1]["accuracy_adaptive_%"])+"%"],
   ["2","mimicry dominant", f2(t6[2]["accuracy_static_%"])+"%", f2(t6[2]["accuracy_adaptive_%"])+"%"],
   ["3","proxy farm arrives", f2(t6[3]["accuracy_static_%"])+"%", f2(t6[3]["accuracy_adaptive_%"])+"%"]],
  [12,34,27,27]));
add(cap("TABLE VII.  Accuracy over a drifting stream."));
add(P(`The frozen model loses ${f2(t6[0]["accuracy_static_%"]-t6[3]["accuracy_static_%"])} points of accuracy by the final block. The monitored model recovers ${f2(t6[3]["accuracy_adaptive_%"]-t6[3]["accuracy_static_%"])} of them, retraining ${R.drift_retrain_points.length} times across the stream. Two caveats belong with this result. Recovery is partial, not complete, because a fresh family is under-represented in the retraining buffer. And the experiment assumes labels arrive promptly; in production, conversion-based labels lag by days, so the detector operates degraded for the duration of that lag. Shortening it is a more valuable engineering target than another point of offline accuracy.`));

add(H2("H","Cost-optimal operating point"));
add(P(`Table VIII sweeps the decision threshold against expected loss under three cost regimes.`));
add(table(
  ["CPC","Cost of a wrong block","τ*","Loss at 0.50","Loss at τ*","Saved per 1M clicks"],
  t5.map(r=>["₹"+r.CPC_INR, r.false_block_multiplier+"× CPC", r.optimal_threshold,
             "₹"+r["cost_per_click_at_0.5"].toFixed(3), "₹"+r.cost_per_click_at_opt.toFixed(3),
             "₹"+Number(r["saving_per_million_clicks_INR"]).toLocaleString('en-IN')]),
  [10,24,10,17,17,22]));
add(cap("TABLE VIII.  Revenue-aware thresholding under three cost regimes."));
add(P(`At ₹${t5[0].CPC_INR} per click with a wrongly blocked customer costing ${t5[0].false_block_multiplier}× the click price, the loss-minimising threshold is ${t5[0].optimal_threshold} rather than 0.50, worth ₹${Number(t5[0]["saving_per_million_clicks_INR"]).toLocaleString('en-IN')} per million clicks. The threshold moves with the cost ratio and not with the classifier, which is the point: the same model has different correct operating points for different advertisers, and reporting a single accuracy figure conceals that.`));

add(H2("I","Latency"));
add(P(`Real-time bidding budgets are measured in milliseconds. Table IX reports measured latency on a single CPU core. The random forest costs ${lat["RandomForest (+EVGF)"][0].batch_latency_ms} ms for a single click but reaches ${lat["RandomForest (+EVGF)"][3].throughput_clicks_per_s.toLocaleString()} clicks per second at batch 1024; the attention network answers a single click in ${lat["CNN+Attention"][0].batch_latency_ms} ms. Both are compatible with micro-batched serving; only the forest is unsuitable for one-at-a-time scoring in a bid path, which is a deployment constraint no accuracy table records.`));
add(table(
  ["Batch size","Random forest, batch (ms)","Random forest, clicks/s","CNN+Attention, batch (ms)","CNN+Attention, clicks/s"],
  lat["RandomForest (+EVGF)"].map((r,i)=>[r.batch_size, r.batch_latency_ms,
      r.throughput_clicks_per_s.toLocaleString(),
      lat["CNN+Attention"][i].batch_latency_ms,
      lat["CNN+Attention"][i].throughput_clicks_per_s.toLocaleString()]),
  [14,24,20,22,20]));
add(cap("TABLE IX.  Inference latency, single CPU core."));

/* ------------------------------- VI ------------------------------------ */
add(H1("VI","Discussion"));
add(P(`Three conclusions follow from the results above.`));
add(P(`First, the reported accuracy of a click-fraud detector is close to uninformative about its deployed value. Eleven of our fourteen models sit within three F1 points, yet the same feature set that produces ${f2(ablB["F1-Score"])} F1 collapses to an ${f2(advLast["ASR_behaviour_only_%"])}% attack-success rate under mimicry, and the same detector that recalls ${f2(inSup.Recall)}% of familiar fraud recalls ${f2(zdSup.Recall)}% of an unfamiliar family. The properties that separate systems are not on the axis the literature reports.`));
add(P(`Second, where a feature is observed matters more than which model consumes it. The gap between a detector an operator can defeat and one it cannot is the gap between browser-reported telemetry and exchange-side counters. This suggests that effort spent on architecture search is misallocated relative to effort spent on server-side instrumentation.`));
add(P(`Third, supervised and novelty detection are complementary rather than competing, and the trade between them is a dial. Setting the false-positive budget β makes the cost of catching unknown fraud explicit and bounded, which is the form in which an advertiser can actually make the decision.`));
add(H2("A","Limitations"));
add(P(`Our results are obtained on a simulator, and a simulator encodes its author's beliefs about how bots behave. We have tried to make those beliefs explicit and adverse to ourselves — overlapping classes, four families, label noise, an attack constrained to what an operator controls — but the generative assumptions remain assumptions, and absolute figures should be read as relative comparisons under a stated model rather than as production estimates. The leave-one-family-out protocol rotates four families, which is enough to show the gain is uneven but not enough to characterise where the boundary between 'variation' and 'new kind' falls. The drift experiment assumes prompt labels. Finally, the mimicry attack is a feature-space attack; a problem-space attack that must also produce a coherent HTTP session would be a stronger test.`));

add(H1("VII","Conclusion"));
add(P(`Ad click-fraud research has converged on a benchmark that rewards recognising fraud it has already seen. We reproduced that benchmark and then measured four things it omits. Entity-velocity graph features raise F1 from ${f2(ablB["F1-Score"])} to ${f2(ablF["F1-Score"])} and cut wrongly blocked visitors by ${((1-ablF.FPR/ablB.FPR)*100).toFixed(0)}%. A denoising autoencoder over legitimate behaviour raises recall on a withheld bot family from ${f2(zdSup.Recall)}% to ${f2(zdCas.Recall)}% for ${f2(inSup["F1-Score"]-inCas["F1-Score"])} F1 on ordinary traffic, and across all four leave-one-family-out folds from ${f2(lofoMeanSup)}% to ${f2(lofoMeanCas)}% — a gain that is real but confined to families unlike those already seen. Under full-fidelity mimicry, server-side features hold attack success to ${f2(advLast["ASR_plus_EVGF_%"])}% where client-side features permit ${f2(advLast["ASR_behaviour_only_%"])}%, and adversarial training reaches ${f2(advLast["ASR_plus_EVGF_advtrain_%"])}%. Drift monitoring recovers ${f2(t6[3]["accuracy_adaptive_%"]-t6[3]["accuracy_static_%"])} accuracy points on a shifting stream, and cost-derived thresholding recovers ₹${Number(t5[0]["saving_per_million_clicks_INR"]).toLocaleString('en-IN')} per million clicks. We suggest that attack-success rate under a constrained adversary, recall on a held-out bot family, and expected loss at the chosen operating point should accompany accuracy in future work on this problem. The full pipeline, simulator and evaluation harness are released with this paper.`));

/* ------------------------------ refs ----------------------------------- */
add(H1("","References"));
const refs = [
 `R. A. Alzahrani, M. Aljabri, and R. M. A. Mohammad, "Ad click fraud detection using machine learning and deep learning algorithms," IEEE Access, vol. 13, pp. 12746–12763, 2025, doi: 10.1109/ACCESS.2025.3532200.`,
 `S. S. Duvvuri and Z. H. Choudhury, "Ad click fraud detection using machine learning and deep learning techniques," SSRN Electronic Journal, 2025, doi: 10.2139/ssrn.5758123.`,
 `"Click fraud detection in online advertising using advanced machine learning models," International Journal of Engineering Research and Science & Technology, 2026.`,
 `"Click fraud detection in online advertising: a comparative study," International Information and Engineering Technology Association, 2025.`,
 `M. Aljabri and R. M. A. Mohammad, "Click fraud detection for online advertising using machine learning," Egyptian Informatics Journal, vol. 24, no. 3, pp. 241–252, 2023.`,
 `A. Batool and Y. Byun, "An ensemble architecture based on deep learning model for click fraud detection in pay-per-click advertisement campaign," IEEE Access, vol. 10, pp. 99234–99249, 2022.`,
 `G. S. Thejas, S. Dheeshjith, S. S. Iyengar, N. R. Sunitha, and P. Badrinath, "A hybrid and effective learning approach for click fraud detection," Machine Learning with Applications, vol. 3, p. 100016, 2021.`,
 `G. S. Thejas, K. G. Boroojeni, K. Chandna, I. Bhatia, S. S. Iyengar, and N. R. Sunitha, "Deep learning-based model to fight against ad click fraud," in Proc. ACM Southeast Conf., 2019, pp. 176–181.`,
 `B. Kirkwood, M. Vanamala, and N. Seliya, "Click fraud detection of online advertising using machine learning algorithms," in Proc. IEEE Int. Conf. Electro Information Technology (EIT), 2024.`,
 `L. Sisodia and D. S. Sisodia, "AI-based techniques for ad click fraud detection and prevention: review and research directions," Journal of Sensor and Actuator Networks, vol. 12, no. 1, p. 4, 2023.`,
 `"Robust AI for financial fraud detection: a hybrid framework for imbalance, drift and adversarial threats," Journal of Theoretical and Applied Electronic Commerce Research, vol. 20, no. 2, p. 121, 2025.`,
 `"Detecting concept drift in financial fraud using temporal graph neural networks," preprint, 2025.`,
 `D. Cartella et al., "Adversarial learning in real-world fraud detection: challenges and perspectives," arXiv:2307.01390, 2023.`,
 `A. Vaswani et al., "Attention is all you need," in Advances in Neural Information Processing Systems, vol. 30, 2017.`,
 `F. T. Liu, K. M. Ting, and Z.-H. Zhou, "Isolation forest," in Proc. IEEE Int. Conf. Data Mining, 2008, pp. 413–422.`,
 `S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in Advances in Neural Information Processing Systems, vol. 30, 2017.`,
 `A. Bifet and R. Gavaldà, "Learning from time-changing data with adaptive windowing," in Proc. SIAM Int. Conf. Data Mining, 2007, pp. 443–448.`,
 `T. Chen and C. Guestrin, "XGBoost: a scalable tree boosting system," in Proc. ACM SIGKDD, 2016, pp. 785–794.`,
];
refs.forEach((r,i)=>add(new Paragraph({
  alignment: AlignmentType.JUSTIFIED, spacing:{after:70},
  indent:{left:convertInchesToTwip(0.32), hanging:convertInchesToTwip(0.32)},
  children:[new TextRun({text:`[${i+1}]\u2003${r}`, font:FONT, size:17})]})));

const doc = new Document({
  creator:"AdShield-X",
  title:"AdShield-X: Entity-Graph Features, Zero-Day Novelty Detection and Revenue-Aware Thresholding for Ad Click Fraud",
  styles:{default:{document:{run:{font:FONT, size:20, color:"000000"}},
    heading1:{run:{color:"000000"}}, heading2:{run:{color:"000000"}}}},
  sections:[{
    properties:{page:{size:{width:12240, height:15840},
      margin:{top:1440,bottom:1440,left:1440,right:1440}}},
    footers:{default:new Footer({children:[new Paragraph({
      alignment:AlignmentType.CENTER,
      children:[new TextRun({children:[PageNumber.CURRENT], font:FONT, size:17})]})]})},
    children: body
  }]
});

Packer.toBuffer(doc).then(b=>{
  fs.writeFileSync('/mnt/user-data/outputs/AdShield-X_Research_Paper.docx', b);
  console.log('written', (b.length/1024).toFixed(1), 'KB');
});
