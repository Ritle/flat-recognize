// Execute the real UI script against a minimal DOM, without browser permissions.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../demo/index.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const nodes = new Map();
function node() {
    return {style: {}, textContent: '', children: [], classList: {toggle() {}, add() {}, remove() {}},
        addEventListener() {}, appendChild(child) { this.children.push(child); }, scrollIntoView() {},
        removeAttribute(name) { delete this[name]; }};
}
let fetches = 0;
const context = vm.createContext({
    document: {
        getElementById(id) { if (!nodes.has(id)) nodes.set(id, node()); return nodes.get(id); },
        createElement: node, querySelectorAll: () => []
    },
    fetch: async () => { fetches++; return {ok: true, json: async () => ({editable: true})}; },
    navigator: {clipboard: {writeText: async () => {}}},
    window: {scrollTo() {}}, setTimeout, Date, console
});
vm.runInContext(script, context);
async function show(status) {
    const allowed = status !== 'invalid';
    await context.showResult({quality: {status, errors: allowed ? [] : ['Invalid geometry']},
        warnings: ['Approximate scale'], stats: {rooms: allowed ? 5 : 0, walls: 18, doors: 5, windows: 6},
        seconds: 12, filename: 'plan.png', job_id: 'sample-job', images: {}, original_url: '/original',
        json_url: allowed ? '/json' : null, download_url: allowed ? '/download' : null});
    assert.equal(nodes.get('downloadButton').hidden, !allowed);
    assert.equal(nodes.get('copyJson').disabled, !allowed);
    assert.equal(nodes.get('jsonPanel').style.display, allowed ? '' : 'none');
    if (!allowed) {
        assert.equal(nodes.get('downloadButton').href, undefined);
        assert.equal(nodes.get('jsonPreview').textContent, '');
        assert.equal(vm.runInContext('currentJson', context), null);
        assert.match(nodes.get('recognitionWarnings').textContent, /Экспорт заблокирован/);
    } else {
        assert.equal(nodes.get('downloadButton').href, '/download');
        assert.match(nodes.get('jsonPreview').textContent, /editable/);
    }
}
(async () => {
    await show('good');
    assert.equal(fetches, 1);
    await show('invalid'); // Prior successful JSON/link must be cleared.
    assert.equal(fetches, 1); // Never fetch a blocked or null JSON URL.
    await context.showResult({quality: {status: 'invalid', errors: []}, warnings: [],
        opening_preflight: {status: 'audited', summary: {examined: 5, suspect: 2},
            wall_geometry: {sampled: 10, suspect: 3}},
        stats: {rooms: 0, walls: 0, doors: 0, windows: 0}, seconds: 1,
        filename: 'plan.png', job_id: 'sample-job', images: {}, original_url: '/original'});
    assert.match(nodes.get('recognitionWarnings').textContent,
        /подозрительных привязок проёмов — 2/);
    assert.match(nodes.get('recognitionWarnings').textContent,
        /подозрительных участков — 3/);
    await show('review'); // Review restores export controls.
    assert.equal(fetches, 2);
    assert.match(nodes.get('recognitionWarnings').textContent, /ручной проверки/);
    nodes.get('tabs').children = [];
    context.setupImages({images: {wall_geometry: '/wall-geometry'}});
    assert.equal(nodes.get('tabs').children[0].textContent, 'Проверка стен');
    nodes.get('tabs').children[0].onclick();
    assert.match(nodes.get('imageCaption').textContent, /Красный — ось стены в проекте/);
    nodes.get('tabs').children = [];
    context.setupImages({images: {recovery: '/recovery'}});
    const tab = nodes.get('tabs').children[0];
    assert.equal(tab.textContent, 'Восстановленные проёмы');
    tab.onclick();
    assert.match(nodes.get('preview').src, /^\/recovery\?t=/);
    assert.match(nodes.get('imageCaption').textContent, /голубой — восстановленное окно/);
    assert.equal(nodes.get('imageCaption').style.display, 'block');
    nodes.get('tabs').children = [];
    context.setupImages({images: {topology_windows: '/topology-windows'}});
    assert.equal(nodes.get('tabs').children[0].textContent, 'Окна по топологии');
    nodes.get('tabs').children[0].onclick();
    assert.match(nodes.get('preview').src, /^\/topology-windows\?t=/);
    assert.match(nodes.get('imageCaption').textContent, /жёлтая линия — возможная ось рамы/);
    assert.match(nodes.get('imageCaption').textContent, /Экспорт такого окна заблокирован/);
    nodes.get('tabs').children = [];
    context.setupImages({images: {interior_windows: '/interior-windows'}});
    assert.equal(nodes.get('tabs').children[0].textContent, 'Отклонённые окна');
    nodes.get('tabs').children[0].onclick();
    assert.match(nodes.get('imageCaption').textContent, /экспорт заблокирован/);
    nodes.get('tabs').children = [];
    context.setupImages({images: {segmentation_original: '/original-pass',
        segmentation_normalized: '/normalized-pass', segmentation_fused: '/fused-pass'}});
    assert.deepEqual(nodes.get('tabs').children.map(tab => tab.textContent),
        ['Сегментация: исходник', 'Сегментация: нормализация', 'Сегментация: fusion']);
    context.setupImages({images: {}, original_url: '/original'});
    assert.equal(nodes.get('imageCaption').style.display, 'none');
    const question = {id: 'opening:8', kind: 'opening', opening_id: 8,
        fingerprint: 'geometry-hash', image_sha256: 'image-hash',
        current: 'door', reasons: ['classification_uncertain'],
        center: [20, 30], image_size: [100, 80]};
    context.renderReviewQuestions({review_questions: [question], applied_decisions: []});
    assert.equal(nodes.get('reviewPanel').style.display, 'block');
    context.renderReviewQuestions({review_questions: [{...question,
        current: 'window', reasons: ['door_arc_missing_jamb']}], applied_decisions: []});
    assert.match(nodes.get('reviewList').children.at(-1).children[1].textContent,
        /похоже на дверь, но контур стены неполный/);
    context.renderReviewQuestions({review_questions: [{...question,
        current: 'window', reasons: ['same_room_window_ambiguous']}], applied_decisions: []});
    assert.match(nodes.get('reviewList').children.at(-1).children[1].textContent,
        /по обе стороны окна распознана одна комната/);
    context.renderReviewQuestions({review_questions: [{...question,
        current: 'window', reasons: ['parallel_window_ambiguous'],
        paired_with: [7]}], applied_decisions: []});
    assert.match(nodes.get('reviewList').children.at(-1).children[1].textContent,
        /параллельным кандидатом/);
    assert.match(nodes.get('reviewList').children.at(-1).children[1].textContent,
        /проём №7/);
    context.renderReviewQuestions({review_questions: [{...question,
        current: 'topology_only', reasons: ['duplicate_parallel_window_layer'],
        paired_with: [7]}], applied_decisions: []});
    assert.match(nodes.get('reviewList').children.at(-1).children[1].textContent,
        /только граница комнаты.*повторный параллельный слой окна/);
    context.renderReviewQuestions({review_questions: [{...question,
        current: 'topology_only', reasons: ['one_sided_window_endcap'],
        paired_with: [7]}], applied_decisions: []});
    assert.match(nodes.get('reviewList').children.at(-1).children[1].textContent,
        /неподтверждённый торец соседнего окна/);
    context.renderReviewQuestions({review_questions: [{...question,
        current: 'topology_only', reasons: ['one_sided_unconfirmed_window']}],
        applied_decisions: []});
    assert.match(nodes.get('reviewList').children.at(-1).children[1].textContent,
        /неподтверждённым косяком/);
    context.renderReviewQuestions({review_questions: [{...question,
        current: 'topology_only', reasons: ['no_jamb_unconfirmed_window']}],
        applied_decisions: []});
    assert.match(nodes.get('reviewList').children.at(-1).children[1].textContent,
        /без опоры стены по краям/);
    context.renderReviewQuestions({review_questions: [{...question,
        current: 'window', reasons: ['topology_recovered_thick_window',
            'thick_window_geometry_unresolved']}], applied_decisions: []});
    assert.match(nodes.get('reviewList').children.at(-1).children[1].textContent,
        /выбор типа их не исправит/);
    context.renderReviewQuestions({review_questions: [{...question,
        reasons: ['opening_exceeds_segment'],
        geometry: {opening_width_cm: 92, wall_length_cm: 50,
            overhang_cm: 45, wall_chain_length_cm: 100}}], applied_decisions: []});
    const row = nodes.get('reviewList').children.at(-1);
    assert.match(row.children[1].textContent, /ширина 92 см/);
    assert.match(row.children[1].textContent, /требует проверки/);
    const select = row.children[2];
    select.value = 'window';
    let submitted;
    context.captureReview = (file, payload) => { submitted = {file, payload}; };
    vm.runInContext('currentFile = {name: "plan.png"}; upload = captureReview;', context);
    nodes.get('applyReviewButton').onclick();
    assert.equal(submitted.file.name, 'plan.png');
    assert.equal(submitted.payload.image_sha256, 'image-hash');
    assert.equal(submitted.payload.answers[0].choice, 'window');
    assert.equal(submitted.payload.answers[0].fingerprint, 'geometry-hash');
    const roomQuestion = {id: 'room:6', kind: 'room', room_id: 6,
        fingerprint: 'room-hash', image_sha256: 'image-hash', current: 'room',
        reasons: ['small_enclosed_area'], center: [40, 40], image_size: [100, 80]};
    context.renderReviewQuestions({review_questions: [roomQuestion], applied_decisions: []});
    const roomRow = nodes.get('reviewList').children.at(-1);
    assert.match(roomRow.children[1].textContent, /частью мебели/);
    roomRow.children[2].value = 'ignore';
    nodes.get('applyReviewButton').onclick();
    assert.equal(submitted.payload.answers[0].room_id, 6);
    assert.equal(submitted.payload.answers[0].choice, 'ignore');
    const partitionQuestion = {id: 'partition:0', kind: 'partition', partition_id: 0,
        fingerprint: 'partition-hash', image_sha256: 'image-hash', current: 'open',
        reasons: ['facing_wall_caps_only'], center: [30, 50], image_size: [100, 80]};
    context.renderReviewQuestions({review_questions: [partitionQuestion], applied_decisions: []});
    const partitionRow = nodes.get('reviewList').children.at(-1);
    assert.match(partitionRow.children[1].textContent, /дверь или сплошная стена не подтверждены/);
    partitionRow.children[2].value = 'separate';
    nodes.get('applyReviewButton').onclick();
    assert.equal(submitted.payload.answers[0].partition_id, 0);
    assert.equal(submitted.payload.answers[0].choice, 'separate');
    partitionQuestion.reasons.push('near_closed_leaf_unconfirmed');
    partitionQuestion.choices = ['door', 'separate', 'open'];
    context.renderReviewQuestions({review_questions: [partitionQuestion], applied_decisions: []});
    assert.match(nodes.get('reviewList').children.at(-1).children[1].textContent,
        /почти закрытое дверное полотно/);
    assert.ok(nodes.get('reviewList').children.at(-1).children[2].children.some(
        option => option.value === 'door'));
    nodes.get('reviewList').children.at(-1).children[2].value = 'door';
    nodes.get('applyReviewButton').onclick();
    assert.equal(submitted.payload.answers[0].choice, 'door');
    const exteriorQuestion = {id: 'exterior:0', kind: 'exterior', exterior_id: 0,
        fingerprint: 'exterior-hash', image_sha256: 'image-hash', current: 'pending',
        center: [40, 20], image_size: [100, 80]};
    context.renderReviewQuestions({review_questions: [exteriorQuestion], applied_decisions: []});
    const exteriorRow = nodes.get('reviewList').children.at(-1);
    assert.match(exteriorRow.children[0].textContent, /внешний контур/);
    assert.match(exteriorRow.children[1].textContent, /пробное замыкание/);
    exteriorRow.children[2].value = 'confirm';
    nodes.get('applyReviewButton').onclick();
    assert.equal(submitted.payload.answers[0].exterior_id, 0);
    assert.equal(submitted.payload.answers[0].choice, 'confirm');
    assert.equal(submitted.payload.answers[0].opening_id, undefined);
    context.renderReviewQuestions({review_questions: [{...exteriorQuestion, current: 'confirm',
        trial: {status: 'accepted', rooms_before: 5, rooms_after: 6}}],
        applied_decisions: [{exterior_id: 0, choice: 'confirm'}]});
    assert.equal(nodes.get('reviewList').children.at(-1).children[2].value, 'confirm');
    assert.match(nodes.get('reviewList').children.at(-1).children[1].textContent, /5 → 6/);
    assert.match(nodes.get('reviewList').children.at(-1).children[1].textContent, /не включена в проект/);
    context.renderClosureCandidates({original_url: '/original', geometry_review_candidates: [
        {closure_index: 2, center: [25, 30], image_size: [100, 80],
            closure_evidence: 'door_leaf', recovery_reason: 'missing_wall_jamb',
            jamb_support: [1, 3], unsupported_edge_count: 3}]});
    assert.equal(nodes.get('closurePanel').style.display, 'block');
    assert.equal(nodes.get('closureList').children.length, 1);
    assert.match(nodes.get('closureList').children[0].children[1].textContent, /1\/3/);
    assert.match(nodes.get('closureList').children[0].children[1].textContent, /3$/);
    assert.equal(nodes.get('closureList').children[0].children.length, 2);
    context.renderClosureCandidates({geometry_review_candidates: []});
    assert.equal(nodes.get('closurePanel').style.display, 'none');
    console.log('UI good → invalid → review and recovery diagnostics: passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
