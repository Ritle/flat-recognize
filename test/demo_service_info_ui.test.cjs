const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../demo/index.html'), 'utf8');
assert.doesNotMatch(html, /\{\{SERVICE_/);
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const loader = script.slice(0, script.indexOf('\nloadServiceInfo();'));

async function run(fetch, initial = 'недоступна') {
    const nodes = new Map([
        ['serviceVersion', {textContent: initial}],
        ['serviceUpdatedAt', {textContent: 'не указана'}],
        ['serviceInfoStatus', {hidden: true}]
    ]);
    const context = vm.createContext({document: {getElementById: id => nodes.get(id)},
        fetch, Date, Intl, Number, Error});
    vm.runInContext(loader, context);
    await context.loadServiceInfo();
    return nodes;
}

(async () => {
    let nodes = await run(async (url, options) => {
        assert.equal(url, '/api/version');
        assert.equal(options.cache, 'no-store');
        return {ok: true, json: async () => ({version: '0.1.0+abc1234', updated_at: '2026-11-13T08:10:00Z'})};
    });
    assert.equal(nodes.get('serviceVersion').textContent, '0.1.0+abc1234');
    assert.equal(nodes.get('serviceUpdatedAt').textContent, '13.11.2026 11:10 МСК');
    assert.equal(nodes.get('serviceInfoStatus').hidden, true);
    for (const fetch of [async () => ({ok: false}), async () => {throw new Error('offline');},
        async () => ({ok: true, json: async () => ({})})]) {
        nodes = await run(fetch);
        assert.equal(nodes.get('serviceInfoStatus').hidden, false);
        assert.equal(nodes.get('serviceVersion').textContent, 'недоступна');
    }
    nodes = await run(async () => ({ok: false}), '0.1.0+rendered');
    assert.equal(nodes.get('serviceVersion').textContent, '0.1.0+rendered');
    assert.equal(nodes.get('serviceInfoStatus').hidden, true);
    nodes = await run(async () => ({ok: true, json: async () => ({version: '<build>', updated_at: null})}));
    assert.equal(nodes.get('serviceVersion').textContent, '<build>');
    assert.equal(nodes.get('serviceUpdatedAt').textContent, 'не указана');
    console.log('Service metadata: static HTML, API date, old server and offline fallback passed');
})().catch(error => {console.error(error); process.exitCode = 1;});
