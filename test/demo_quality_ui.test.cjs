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
    assert.equal(tab.textContent, 'Восстановленные двери');
    tab.onclick();
    assert.match(nodes.get('preview').src, /^\/recovery\?t=/);
    assert.match(nodes.get('imageCaption').textContent, /найденное полотно/);
    assert.equal(nodes.get('imageCaption').style.display, 'block');
    context.setupImages({images: {}, original_url: '/original'});
    assert.equal(nodes.get('imageCaption').style.display, 'none');
    console.log('UI good → invalid → review and recovery diagnostics: passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
