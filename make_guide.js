const fs = require('fs');
const d = require('docx');
const {Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
       Table, TableRow, TableCell, WidthType, ShadingType, PageNumber, Footer,
       convertInchesToTwip, BorderStyle} = d;

const F = "Calibri";
const CW = 9360;
const INK = "1B2A38", MUT = "5C6E7C", ACC = "23566E", RED = "9E2B25", GRN = "1D6B4F";

const P = (t, o={}) => new Paragraph({
  alignment: o.align || AlignmentType.LEFT,
  spacing: {after: o.after ?? 130, line: o.line ?? 264},
  indent: o.indent,
  children: (Array.isArray(t)?t:[t]).map(x =>
    typeof x === 'string'
      ? new TextRun({text:x, font:F, size:o.size||21, bold:o.b, italics:o.i,
                     color:o.c||INK})
      : x)
});
const R = (t,o={}) => new TextRun({text:t, font:F, size:o.size||21, bold:o.b,
                                   italics:o.i, color:o.c||INK});

const H1 = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_1, spacing:{before:320, after:150},
  border:{bottom:{style:BorderStyle.SINGLE, size:6, color:"C9D4DC", space:6}},
  children:[new TextRun({text:t, font:F, size:28, bold:true, color:INK})]});
const H2 = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_2, spacing:{before:230, after:110},
  children:[new TextRun({text:t, font:F, size:23, bold:true, color:ACC})]});
const H3 = (t) => new Paragraph({
  spacing:{before:170, after:80},
  children:[new TextRun({text:t, font:F, size:21, bold:true, color:INK})]});

const BULLET = (t, o={}) => new Paragraph({
  bullet:{level: o.level||0}, spacing:{after:80, line:264},
  children:(Array.isArray(t)?t:[t]).map(x=> typeof x==='string'
    ? new TextRun({text:x, font:F, size:21, color:o.c||INK, bold:o.b}) : x)});

const QUOTE = (t) => new Paragraph({
  spacing:{before:90, after:150, line:270},
  indent:{left:convertInchesToTwip(0.28)},
  border:{left:{style:BorderStyle.SINGLE, size:12, color:ACC, space:10}},
  children:[new TextRun({text:t, font:F, size:21, italics:true, color:INK})]});

function table(headers, rows, widths, opt={}){
  const tot = widths.reduce((a,b)=>a+b,0);
  const cols = widths.map(w=>Math.round(w/tot*CW));
  const cell=(txt,o={})=>new TableCell({
    width:{size:cols[o.i],type:WidthType.DXA},
    shading:o.sh?{type:ShadingType.CLEAR,fill:o.sh,color:"auto"}:undefined,
    margins:{top:70,bottom:70,left:100,right:100},
    children:String(txt).split("\n").map(line=>new Paragraph({
      spacing:{after:0,line:250}, alignment:o.a||AlignmentType.LEFT,
      children:[new TextRun({text:line, font:F, size:o.s||19, bold:o.b,
                             color:o.c||INK})]}))});
  return new Table({columnWidths:cols, width:{size:CW,type:WidthType.DXA},
    rows:[
      new TableRow({tableHeader:true, children:headers.map((h,i)=>
        cell(h,{i,b:true,sh:"E7ECF0"}))}),
      ...rows.map(r=>new TableRow({children:r.map((v,i)=>
        cell(v,{i, c: opt.color? opt.color(v,i) : undefined}))}))
    ]});
}
const cap = t => new Paragraph({spacing:{before:60, after:180},
  children:[new TextRun({text:t, font:F, size:18, color:MUT, italics:true})]});

const body=[]; const add=(...x)=>body.push(...x);

/* ======================= cover ======================= */
add(new Paragraph({spacing:{after:60},
  children:[R("AdShield-X", {size:44, b:true})]}));
add(new Paragraph({spacing:{after:40},
  children:[R("Ad Click Fraud Detection — Speaker Guide", {size:26, c:ACC})]}));
add(new Paragraph({spacing:{after:340},
  children:[R("Ee project ni evariki ayina — friend, faculty, panel, reviewer — simple ga, confident ga ela explain cheyyalo full script.", {size:21, c:MUT})]}));

add(H1("Ee document ni ela vaadali"));
add(P("Idhi mottham chadavaalsina avasaram ledu. Nee situation ni batti section pick chesuko:"));
add(BULLET([R("Evaro casual ga adigaru", {b:true}), R(" → Section 1 (30 seconds)")]));
add(BULLET([R("Friend ki / classmate ki explain chestunnav", {b:true}), R(" → Section 1 + 3 (analogies)")]));
add(BULLET([R("Faculty ki, review meeting lo", {b:true}), R(" → Section 2 + 4 (demo walkthrough)")]));
add(BULLET([R("Final panel / viva / conference", {b:true}), R(" → mottham, mukhyanga Section 6 (Q&A) and 7 (traps)")]));
add(P("Chivarlo one-page cheat sheet undi — daanni print chesi jeb lo pettuko."));

/* ======================= 1 ======================= */
add(H1("1 · 30-second answer"));
add(P("Evaraina \"nee project enti?\" ani adigithe, idhi cheppu. Ekkuva cheppaku."));
add(QUOTE("Google, Facebook laanti platforms lo advertiser prati click ki paisalu isthadu. Kaani aa clicks lo chala clicks bots ivi — real people kaadu. Naa project aa fake clicks ni pattukuntundi. Idhi already unna research kanna difference enti ante — anduru \"97% accuracy vachindi\" ani aagipotharu. Nenu adigina question veru: kotta rakam bot vaste? Bot manaki telisi thana behaviour ni maarchukunte? And asalu entha rupayalu save avuthunnayi? Aa moodintiki naa daggara measured answers unnayi."));
add(P([R("Idhi enduku pani chestundi: ", {b:true}), R("nuvvu \"nenu ML model build chesa\" ani cheppaledu — andaru adhe chestaru. Nuvvu \"nenu vere question adiganu\" ani cheppav. Adhe curiosity create chestundi.")]));

/* ======================= 2 ======================= */
add(H1("2 · 2-minute answer"));
add(P("Faculty ki, ledha interest chupinchina evariki ayina. Naalugu steps lo cheppu — order maarchaku."));

add(H3("Step 1 — Problem"));
add(QUOTE("Pay-per-click ante: nenu advertiser, naa ad ni oka website lo pedatha. Evaraina click cheste aa website owner ki nenu ₹18 istha. Ippudu aa website owner oka script raasi thana ad ni thane 10,000 saarlu click cheste? Naa budget khaali, thana account full. Idhi $80+ billion problem worldwide."));
add(P("Konchem pause ivvu. Ee point andariki artham avvali."));

add(H3("Step 2 — Ippudu anduru em chestunnaru"));
add(QUOTE("Ee problem meeda already chala papers unnayi. Prati okkati same pani chestundi: click data teesuko, 13-14 ML models train cheyyi, accuracy table print cheyyi, \"98% vachindi\" ani cheppu. Nenu kuda modhata adhe reproduce chesa — same 14 models. Naa daggara 97.44% vachindi."));

add(H3("Step 3 — Aa approach lo problem enti"));
add(P("Idhi mukhyamaina point. Nemmadiga cheppu."));
add(QUOTE("Kaani naa 14 models lo 11 models 3 marks lopala unnayi — andaru dadapu okelaage. Appudu \"best model edhi?\" ani accuracy table cheppadhu. Inko problem: aa accuracy anedhi model ki already telisina bots meeda vachina score. Kotta bot vaste? Test evaru cheyyaledu."));
add(QUOTE("So nenu test chesa. Oka kotta type bot ni training nunchi mottham theesesi, tarvata aa bot ni pattukogaldha ani chusa. Normal model 0.56% matrame pattukundi — ante 100 lo 99 bots ni advertiser ki bill chesindi. Accuracy table lo idhi ekkada kanipinchadu."));

add(H3("Step 4 — Manam em chesam"));
add(QUOTE("Nenu 5 vishayalu add chesa. Rendintini matrame cheppali ante: okati, click ni okkatiga chudakunda \"ee IP nunchi gantalo enni clicks vachayi\" ani network level lo chusa — endukante bot thana browser ni fake cheyagaladu kaani mana server counters ni fake cheyaledu. Rendu, \"fraud ela untundo\" nerchukovadam kaakunda \"nijam manishi ela untado\" nerchukune second model pettanu. Aa kotta bot ni normal model 0.56% pattukunte, idhi 83.56% pattukundi."));

/* ======================= 3 ======================= */
add(H1("3 · Analogies — technical word vaadakunda cheppadam"));
add(P("Evarikaina, even non-technical person ki, ee analogies tho artham avutundi. Ivi bagaa gurthu pettuko — demo lo ivi vaadithe impact ekkuva."));

add(H2("Problem — Shop analogy"));
add(QUOTE("Nee daggara shop undi. Oka salesman ki \"prati customer teesukocchinanduku ₹18\" ani commission isthunnav. Aa salesman thana friends ni pilichi shop lo tippi commission teesukuntunnadu. Ippudu shop lo nijamaina customer evaru, fake evaru ani nuvvu ela kanukkuntav?"));

add(H2("Contribution 1 — EVGF (graph features)"));
add(QUOTE("Oka manishi mukham chusi \"idhi nijam customer aa?\" ani cheppadam kashtam — bot bagaa acting cheyagaladu. Kaani nee shop lo unna register chudu: \"ee okka phone number nunchi gantalo 40 orders vachayi\" ante adhi fake. Bot thana behaviour ni fake cheyagaladu — mouse move chesinattu, page scroll chesinattu. Kaani ee register manadhi. Daanni fake cheyalante, thana click speed thaggincukovali. Speed thagginchite thana sampadana thaggutundi. Ante fake cheyyadaniki daaniki cost padutundi."));
add(P([R("Punch line: ", {b:true}), R("\"Bot browser ni control chestundi. Manam server ni control chestam. Detection ni server side meeda build cheste bot evade cheyaledu.\"")]));

add(H2("Contribution 2 — Zero-day cascade (autoencoder)"));
add(QUOTE("Bank lo fake notes pattukovadaniki rendu paddhatulu unnayi. Modhatidhi — \"ippati varaku dorikina fake notes anni ela unnayo\" nerchukovadam. Idhi baagane pani chestundi, kaani kotta rakam fake note vaste fail avutundi, endukante adhi list lo ledu. Rendodhi — \"nijam note ela untundo\" perfect ga nerchukovadam. Appudu ekkadi nunchi kotta note vachina \"idhi nijam note laaga ledu\" ani cheppochu. Naa project lo rendo paddhati pettanu."));
add(P([R("Concrete example ivvu: ", {b:true}), R("\"Mana autoencoder oka vishayam gamaninchindi — desktop computer nunchi vachina click lo touch events unnayi. Desktop ki touch screen ledu kada? Prati value separate ga chuste normal ga undi. Kaani combination impossible. Adhe pattukundi.\"")]));

add(H2("Contribution 3 — Adversarial attack"));
add(QUOTE("Exam lo copy kottadam laantidhi. Student handwriting maarchukogaladu, answer style maarchukogaladu, dress maarchukogaladu — avi anni thana chethilo unnayi. Kaani attendance register lo entry, hall ticket number — avi teacher chethilo. Manam detection ni teacher chethilo unna vaatipai build cheste, student entha try chesina evade cheyaledu."));
add(P([R("Number tho close cheyyi: ", {b:true}), R("\"Browser data meeda matrame detector build cheste bot 89% success rate tho escape avutundi. Mana graph features tho 9.5% ki padipoyindi.\"")]));

add(H2("Contribution 4 — Cost-optimal threshold"));
add(QUOTE("Airport security laantidhi. Prati okkarini agichi full check cheste terrorist dorukutadu — kaani 1000 mandi flight miss avutaru. Evarini agakapothe andaru time ki vellotharu — kaani risk. So \"50-50\" ane default cut point sarikaadu. Rendu tappula lo edhi ekkuva nashtam ani chusi decide cheyyali. Mana case lo: nijam customer ni block cheyyadam, bot click ki paisalu ivvadam kanna 6 rethlu ekkuva nashtam. Anduke mana cut point 0.63, 0.50 kaadu. Aa okka change ₹1,20,000 per 10 lakh clicks save chestundi."));

add(H2("Contribution 5 — Drift monitor"));
add(QUOTE("Chor kotta technique nerchukunte, police kuda nerchukovali. Kaani police prati month calendar chusi training cheyyakoodadhu — waste. Chor maarinappudu maaru. Mana system thana sonta mistakes ni chusukuntundi; mistakes penchite \"aha, evaro technique maarcharu\" ani telusukuni thanantata thanu malli nerchukuntundi."));

add(H2("Leave-one-family-out — final test"));
add(QUOTE("Exam lo 4 chapters unnayi anuko. Prati saari oka chapter ni pakkana pettesi, migilina 3 chapters matrame chadivi, tarvata pakkana pettina chapter meeda exam raayinchadam. Naalugu saarlu. Adhe nenu 4 bot types tho chesa."));

add(H2("Accuracy table enduku useless"));
add(QUOTE("14 mandi students ki exam pettanu, andaru 96-97 marks techukunnaru. Ippudu \"best student evaru?\" ani ela cheptav? Cheppalevu. Kaani vaalaki syllabus lo lene kotta question adigithe — okadu 4 marks, inkokadu 88 marks techukunnadu. Aa question adagakapothe aa difference nuvvu eppatiki chudalevu. Naa project cheppedi adhe: sarina question adagandi."));

/* ======================= 4 ======================= */
add(H1("4 · Demo walkthrough — screen by screen script"));
add(P("Demo HTML open chesi, ee order lo velthe 6-8 minutes lo full story chepesthav. Prati step lo em click cheyyalo, em cheppalo ichanu."));

add(H2("Step 1 — Architecture diagram (30 sec)"));
add(P([R("Cheyyi: ", {b:true}), R("\"How a click gets judged\" section ki scroll cheyyi.")]));
add(P([R("Cheppu: ", {b:true}), R("\"Left nunchi right ki chudandi. Thella boxes anni already unna research lo unnayi. Number unna boxes nenu add chesinavi. Erupu dashed line chudandi — adhi bot ekkadaki cheragaladho. EVGF box daggara ✕ mark undi — akkadaki cheragadu. Pacchani line system thanantata thanu malli nerchukune loop.\"")]));

add(H2("Step 2 — Real visitor (30 sec)"));
add(P([R("Cheyyi: ", {b:true}), R("Pai buttons lo \"Real visitor 1\" click cheyyi.")]));
add(P([R("Cheppu: ", {b:true}), R("\"Idhi mana test data nunchi teesina nijamaina click. Green stamp — legitimate. Fraud risk 2-3%. Kinda reasons chudandi — 'consistent with a real visitor'.\"")]));

add(H2("Step 3 — Crude bot (30 sec)"));
add(P([R("Cheyyi: ", {b:true}), R("\"Crude bot 1\" click cheyyi.")]));
add(P([R("Cheppu: ", {b:true}), R("\"Ippudu red stamp, 95%+ risk. Reasons English lo unnayi: mouse asalu kadalledu, clicks anni exact same gap tho vachayi. Idhi simple bot — evaraina pattukuntaru.\"")]));

add(H2("Step 4 — Mimicry bot (1 min) — ikkada EVGF chupinchu"));
add(P([R("Cheyyi: ", {b:true}), R("\"Mimicry bot 1\" click cheyyi. Reasons list chupinchu.")]));
add(P([R("Cheppu: ", {b:true}), R("\"Idhi smart bot. Mouse kadipindi, scroll chesindi, timing kuda random ga pettindi — behaviour antha manishi laage undi. Kaani reasons chudandi: 'abnormal click volume from this IP', 'publisher traffic sits on very few IPs'. Bot thana browser ni fake chesindi, kaani mana server counters ni fake cheyaledu. Idhe naa modhati contribution live lo.\"")]));

add(H2("Step 5 — Proxy farm (2 min) — demo lo best moment"));
add(P([R("Cheyyi: ", {b:true}), R("\"Proxy farm (unseen) 1\" click cheyyi. Amber box vastundi — daanni chupinchu.")]));
add(P([R("Cheppu: ", {b:true}), R("\"Idhi mottham kotta rakam bot. Training lo idhi asalu ledu. Amber box chadavandi — normal supervised model dheeniki 1.2% risk ichindi. Ante 'idhi nijam manishi, bill chesukondi' ani cheppindi. Kaani mana second model — nijam manishi ela untado nerchukunna model — 'idhi nijam session laaga reconstruct avvatledhu' ani cheppi escalate chesindi.\"")]));
add(P([R("Pause chesi idhi cheppu: ", {b:true}), R("\"Ee okka click meeda ₹18. Ilanti 10 lakhs clicks vaste ₹1.8 crores. Normal model antha vadilesedi.\"")]));

add(H2("Step 6 — Sliders (30 sec)"));
add(P([R("Cheyyi: ", {b:true}), R("\"Pointer moves\" slider ni thaggichu, tarvata \"Clicks from this IP, 1h\" ni penchu.")]));
add(P([R("Cheppu: ", {b:true}), R("\"Idhi recorded video kaadu — live model. Nenu values maarcheste risk maarutundi. Meeru kuda try cheyyandi.\"")]));

add(H2("Step 7 — Leave-one-family-out table (1.5 min)"));
add(P([R("Cheyyi: ", {b:true}), R("\"Every family takes a turn as the unknown\" table chupinchu.")]));
add(P([R("Cheppu: ", {b:true}), R("\"Ikkada nenu naa sonta claim ni test chesa. 4 rakala bots unnayi. Prati saari okadanni thoisi, migilina 3 tho train chesi, thoisina daani meeda test chesa. Modhati rendu rows lo gain dadapu ledu. Chivari row lo 4.94% nunchi 87.75% ki vellindi.\"")]));
add(P([R("Ee line tho close cheyyi (chala important): ", {b:true}), R("\"Idhi naa claim ni chinnadi chesindi, kaani nijam chesindi. Naa second model 'anni kotta fraud' ni pattukoledhu — 'kotta rakam fraud' ni pattukundi. Aa difference nenu paper lo raasa, dachaledhu.\"")]));

add(H2("Step 8 — Adversarial chart (1 min)"));
add(P([R("Cheppu: ", {b:true}), R("\"X-axis meeda bot entha bagaa manishi ni copy chestundo. Erupu line — browser data meeda matrame detector. Bot bagaa copy cheste 89% escape avutundi. Pacchani line — mana graph features tho. 9.5% matrame. Ante robustness anedhi model dhi kaadu, feature ekkadi nunchi vachindo daani dhi.\"")]));

add(H2("Step 9 — Cost curve (1 min) — money tho close cheyyi"));
add(P([R("Cheppu: ", {b:true}), R("\"Chivarlo — anduru 0.50 cut point vaadutharu, adhi accuracy ki best. Kaani nijam customer ni block cheyyadam ekkuva nashtam. Cost curve chuste minimum 0.63 daggara undi. Aa okka number maarchite 10 lakh clicks ki ₹1,20,000 save avutundi. Naa project chivari answer accuracy kaadu — rupayalu.\"")]));

/* ======================= 5 ======================= */
add(H1("5 · Gurthu pettukovalsina numbers"));
add(P("Ee 8 numbers matrame chalu. Migilinavi document lo unnayi."));
add(table(
  ["Number","Idhi enti","Ekkada vaadali"],
  [["97.44%","Best model (XGBoost) accuracy","Baseline reproduce chesanu ani chepputhunnappudu"],
   ["11 / 14","14 models lo 11 models 3 F1 points lopala","\"Accuracy table useless\" point ki"],
   ["0.56% → 83.56%","Kotta bot family meeda recall — normal model vs manadhi","Demo lo main moment"],
   ["65.61% → 86.61%","4 folds average (leave-one-family-out)","Full claim, honest version"],
   ["89% → 9.5% → 1.5%","Attack success: browser-only / +graph / +adversarial training","Robustness point"],
   ["105 → 36","Tappuga block ayina nijam customers (EVGF valla)","Business impact"],
   ["₹1,20,000","10 lakh clicks ki save ayye money","Chivari close"],
   ["4.1×","CNN+Attention XGBoost kanna entha ekkuva real customers block chestundo","Base paper ni question cheyyadaniki"]],
  [22,42,36]));

/* ======================= 6 ======================= */
add(H1("6 · Questions and answers"));
add(P("Ivi eppudaina adugutharu. Answers ready ga pettuko."));

add(H3("\"Idhi already unna project ye kada?\""));
add(P("\"Baseline avunu — nenu deliberately same 14 models reproduce chesa, endukante compare cheyyalante same scale kavali. Kaani aa taruvata 5 kotta layers add chesa, and mukhyanga 4 kotta experiments add chesa. Old project 'accuracy entha?' ani adugutundi. Nenu 'kotta bot vaste?', 'bot maarithe?', 'entha money?' ani adiganu. Aa naalugu answers old project lo ledu.\""));

add(H3("\"Dataset real kaadu kada? Simulator enduku?\""));
add(P("\"Base paper vaadina dataset Veracity Trust Network nunchi — adhi paid, and vaallu publish cheyyaledu. Ante aa paper ni evaru reproduce cheyyaleru. So nenu simulator raasa, kaani deliberately kashtam ga: 4 rakala bots, mobile users ki genuine overlap — mobile lo mouse move undadu kada, adhe bot signature laaga untundi — and 2% labels tappu. Naa assumptions anni code lo open ga unnayi, evaraina chudochu. Original paper numbers kanna idhi ekkuva transparent. Real CSV isthe same pipeline adhe pani chestundi — okka command.\""));
add(P([R("Idhi kuda cheppu (honest ga undadam better): ", {b:true}), R("\"Naa absolute numbers ni production estimates ga chudavaddhu — comparisons ga chudandi. Aa limitation paper lo raasa.\"")]));

add(H3("\"Mee accuracy base paper (99%) kanna thakkuva kada?\""));
add(P("\"Avunu, and adhi expected. Vaalla data lo bots easy ga separate ayyevi. Naa data lo mobile humans and bots overlap avutaru, plus 2% label noise undi. Nenu accuracy penchadaniki try cheyyaledu — accuracy anedhi sarina metric kaadu ani chupinchadaniki try chesa. 11 models 3 points lopala unnayi, kaani okkati kotta bot meeda 0.56%, inkokati 87%. Aa difference accuracy lo kanipinchadu.\""));

add(H3("\"CNN+Attention base paper lo best ani unnadi, mee daggara worst enduku?\""));
add(P("\"Idhi naa results lo interesting finding. 1-D convolution anedhi pakka pakka columns ki sambandham undi ani assume chestundi — image lo pakka pixels ki sambandham unnattu. Kaani click record lo column order arbitrary. 'session_duration' pakkana 'browser' undadaniki reason ledu. Naa daggara AUC lo adhi almost best (97.32), kaani deployed threshold daggara 142 real customers ni block chestundi — XGBoost 35 matrame. Ranking quality and usable operating point rendu veru. Nenu ee finding ni paper lo raasa.\""));

add(H3("\"Mee cascade anni kotta fraud ni pattukuntundha?\""));
add(P([R("Ikkada overclaim cheyyaku. ", {b:true, c:RED}), R("\"Kaadu, and adhe naa leave-one-family-out result cheppedi. 4 folds lo 3 folds lo gain almost ledu — endukante aa families already unna families ki variations. Okka fold lo matrame peddha gain — akkada withheld family structurally veru. So correct claim idhi: cascade 'kotta rakam' fraud ni pattukuntundi, 'kotta instance' ni kaadu. Adhi naa paper lo raasina boundary.\"")]));

add(H3("\"Idhi production lo pettochha?\""));
add(P("\"Konni parts avunu — EVGF layer O(1) counters, single thread meeda second ki 1,792 clicks. Latency kuda measure chesa. Kaani rendu gaps unnayi: okati, real production lo labels 2-3 rojula tarvata vastayi (conversion signals nunchi), naa drift experiment labels ventane vastayi ani assume chestundi. Rendu, naa attack feature-space lo undi — real attacker complete HTTP session generate cheyyali. Rendu limitations paper lo unnayi.\""));

add(H3("\"Deep learning vaadaledha? TensorFlow ledu kada?\""));
add(P("\"Vaadanu — DNN, CNN, RNN, CNN+Attention, plus autoencoder. Kaani nenu 200-line automatic differentiation engine NumPy lo raasa, and gradient-check chesa finite differences tho. Ante ee project GPU lekunda, TensorFlow install cheyyakunda, e laptop lo ayina run avutundi. Aa gradient check command README lo undi — ippude run chesi chupinchagalanu.\""));

add(H3("\"Mee contribution lo edhi most important?\""));
add(P("\"Rendu. Okati — robustness anedhi model architecture dhi kaadu, feature ekkadi nunchi vachindo daani dhi. Browser data bot chethilo, server counters mana chethilo. Rendu — supervised model kotta rakam fraud ki gudddidhi, and daaniki fix 'inko better classifier' kaadu, 'fraud ela untundo kaakunda nijam ela untundo nerchukune model'.\""));

/* ======================= 7 ======================= */
add(H1("7 · Ivi cheppaku"));
add(P("Ee 5 traps lo padaku. Prati okka dhaniki correct version ichanu."));
add(table(
  ["Cheppaku","Cheppu"],
  [["\"Naa model 99% accuracy\"","\"97.44%, kaani accuracy ee problem ki sarina metric kaadu — anduke nenu 4 vere experiments chesa.\""],
   ["\"Naa system anni kotta bots ni pattukuntundi\"","\"Structurally kotta bots ni pattukuntundi. Familiar bots ki variations ayithe supervised model already handle chestundi.\""],
   ["\"Real dataset meeda test chesa\"","\"Simulator meeda. Base paper dataset paid and unpublished. Naa assumptions anni code lo open ga unnayi.\""],
   ["\"Idhi production ready\"","\"Research prototype. Latency measure chesa, kaani label delay and problem-space attacks inka pending.\""],
   ["\"Base paper thappu\"","\"Base paper numbers ni nenu question cheyyaledu. Nenu question chesindi aa protocol em measure chestundo anedhi.\""]],
  [40,60]));

/* ======================= 8 ======================= */
add(new Paragraph({children:[new d.PageBreak()]}));
add(H1("Cheat sheet — okka page"));
add(P("Print chesi jeb lo pettuko.", {c:MUT}));

add(H3("Oka line lo"));
add(P("\"Fake ad clicks ni pattukune project. Difference: anduru accuracy chusthaaru, nenu kotta bot, adapting bot, and money — ee moodu chusanu.\""));

add(H3("Demo order"));
add(P("Architecture → Real visitor → Crude bot → Mimicry bot (EVGF point) → Proxy farm (best moment) → Sliders → LOFO table (honest boundary) → Adversarial chart → Cost curve (close)."));

add(H3("5 contributions, okko line"));
add(BULLET("EVGF — click ni okkatiga kaakunda IP/device/publisher network lo chudadam. Bot browser fake cheyagaladu, mana server counters kaadu."));
add(BULLET("Zero-day cascade — 'fraud ela untundo' kaakunda 'nijam manishi ela untado' nerchukune second model."));
add(BULLET("Adversarial test — bot manishi ni copy cheste em avutundo measure cheyyadam."));
add(BULLET("Cost threshold — 0.50 kaadu, 0.63. Real customer ni block cheyyadam 6× ekkuva nashtam."));
add(BULLET("Drift monitor — calendar chusi kaadu, chor maarinappudu manam maaradam."));

add(H3("6 numbers"));
add(P("97.44% · 11 of 14 · 0.56%→83.56% · 65.61%→86.61% · 89%→9.5%→1.5% · ₹1,20,000"));

add(H3("Rendu strongest lines"));
add(QUOTE("\"Robustness anedhi model dhi kaadu — feature ekkadi nunchi vachindo daani dhi.\""));
add(QUOTE("\"Naa leave-one-family-out result naa sonta claim ni chinnadi chesindi. Nenu daanni dachaledhu, paper lo raasa. Adhe naa project lo naaku nachina part.\""));

add(H3("Panel lo confidence kosam"));
add(P("Nuvvu answer cheyyalenidhi vaste — \"adhi naa paper lo limitations section lo undi\" ani cheppu, and nijam ga akkada undi. Guess cheyyaku. \"Teliyadu, kaani ela test cheyyalo cheppagalanu\" anedhi panel ki bagaa nachutundi."));

/* ======================= build ======================= */
const doc = new Document({
  creator:"AdShield-X",
  title:"AdShield-X Speaker Guide",
  styles:{default:{document:{run:{font:F, size:21, color:INK}},
    heading1:{run:{color:INK}}, heading2:{run:{color:ACC}}}},
  sections:[{
    properties:{page:{size:{width:12240,height:15840},
      margin:{top:1300,bottom:1300,left:1400,right:1400}}},
    footers:{default:new Footer({children:[new Paragraph({
      alignment:AlignmentType.CENTER,
      children:[new TextRun({children:[PageNumber.CURRENT], font:F, size:17,
                             color:MUT})]})]})},
    children: body
  }]
});
Packer.toBuffer(doc).then(b=>{
  fs.writeFileSync('/mnt/user-data/outputs/AdShield-X_Explanation_Guide.docx', b);
  console.log('written', (b.length/1024).toFixed(1),'KB');
});
