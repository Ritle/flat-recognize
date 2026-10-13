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
    await show('review'); // Review restores export controls.
    assert.equal(fetches, 2);
    assert.match(nodes.get('recognitionWarnings').textContent, /ручной проверки/);
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
    assert.match(nodes.get('imageCaption').textContent, /Требует проверки/);
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
    const question = {opening_id: 8, fingerprint: 'geometry-hash', image_sha256: 'image-hash',
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
    console.log('UI good → invalid → review and recovery diagnostics: passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
