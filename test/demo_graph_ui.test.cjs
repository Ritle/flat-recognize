const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('demo/index.html', 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const nodes = new Map();
function node() {
    const listeners = {};
    return {style: {}, children: [], value: '', hidden: false, width: 200, height: 100,
        classList: {toggle() {}, add() {}, remove() {}},
        set textContent(value) { this._text = value; this.children = []; },
        get textContent() { return this._text || ''; },
        addEventListener(name, fn) { listeners[name] = fn; },
        dispatch(name, event) { listeners[name](event); },
        appendChild(child) { this.children.push(child); }, scrollIntoView() {}, removeAttribute() {},
        getBoundingClientRect() { return {left: 0, top: 0, width: 100, height: 50}; },
        setPointerCapture() {}, getContext() { return {drawImage() {}, fillRect() {}, strokeRect() {}}; }};
}
class FormDataMock {
    constructor() { this.data = {}; }
    append(key, value) { this.data[key] = value; }
}
class ImageMock {
    constructor() { this.naturalWidth = 200; this.naturalHeight = 100; }
    set src(value) { this.onload(); }
}
const calls = [];
const question = {id: 'gap:test', kind: 'gap', choices: ['door', 'undecided'], image_sha256: 'hash'};
let resolveFetch;
const context = vm.createContext({
    document: {getElementById(id) { if (!nodes.has(id)) nodes.set(id, node()); return nodes.get(id); },
               createElement: node, querySelectorAll: () => []},
    fetch: async (url, options) => {
        if (url === '/api/version') return {ok: true, json: async () => ({version: 'test'})};
        calls.push({url, data: options.body.data});
        const data = {job_id: calls.length === 1 ? 'graph-1' : 'graph-2', pipeline: 'graph-review',
            quality: {status: 'invalid', errors: []}, stats: {rooms: 0, walls: 0, doors: 0, windows: 0},
            images: {graph_questions: '/questions'}, original_url: '/original', seconds: 1,
            warnings: [], review_questions: calls.length === 1 ? [question] : [], image_sha256: 'hash'};
        return {ok: true, json: async () => data};
    },
    FormData: FormDataMock, Image: ImageMock, navigator: {clipboard: {}}, window: {scrollTo() {}},
    setTimeout, Date, console
});
vm.runInContext(script, context);
(async () => {
    await context.showResult({job_id: 'source-job', pipeline: 'v2', quality: {status: 'invalid', errors: []},
        stats: {rooms: 7, walls: 93, doors: 1, windows: 24}, images: {}, original_url: '/original',
        filename: 'plan.png', warnings: [], seconds: 1});
    assert.equal(nodes.get('resultTitle').textContent, 'Экспорт заблокирован');
    assert.equal(nodes.get('windowsLabel').textContent, 'Кандидатов в окна');
    assert.equal(nodes.get('doorsLabel').textContent, 'Кандидатов в двери');
    assert.equal(nodes.get('windowsCount').textContent, 24);
    await nodes.get('graphStartButton').onclick();
    assert.equal(calls.length, 0); // Scale is required.
    nodes.get('graphScale').value = '3.9';
    nodes.get('graphScopeButton').onclick();
    const canvas = nodes.get('graphScopeCanvas');
    canvas.dispatch('pointerdown', {clientX: 10, clientY: 10, pointerId: 1});
    canvas.dispatch('pointerup', {clientX: 30, clientY: 20, pointerId: 1});
    await nodes.get('graphStartButton').onclick();
    assert.equal(calls[0].url, '/api/jobs/source-job/graph-review');
    assert.equal(calls[0].data.scale, '3.9');
    assert.deepEqual(JSON.parse(calls[0].data.scope).exclude_polygons,
                     [[[20,20],[60,20],[60,40],[20,40]]]);
    const select = nodes.get('graphReviewList').children[0].children[1];
    assert.deepEqual(select.children.map(option => option.value), ['', 'door', 'undecided']);
    select.value = 'door';
    await nodes.get('graphApplyButton').onclick();
    assert.equal(calls[1].url, '/api/jobs/graph-1/graph-review');
    assert.deepEqual(JSON.parse(calls[1].data.decisions).answers, [{id: 'gap:test', choice: 'door'}]);
    assert.equal(nodes.get('graphReviewPanel').style.display, 'none');
    nodes.get('tabs').children = [];
    context.setupImages({images: {graph_export: '/export'}});
    assert.equal(nodes.get('tabs').children[0].textContent, 'Геометрия графа');
    nodes.get('tabs').children[0].onclick();
    assert.match(nodes.get('imageCaption').textContent, /оси стен в JSON/);
    context.renderGraphQuestions({pipeline:'graph-review', review_questions:[{...question,
        kind:'window', reason:'possible_door_leaf_near_window', choices:['door','window','undecided'],
        leaf_context:{near_tip_cross_strokes:[{}]}}]});
    assert.match(nodes.get('graphReviewList').children.at(-1).children[0].textContent, /диагональная линия/);
    assert.match(nodes.get('graphReviewList').children.at(-1).children[0].textContent, /может быть мебель/);
    context.renderGraphQuestions({pipeline:'graph-review', review_questions:[{...question,
        kind:'partition', choices:['solid_wall_with_door','undecided']}]});
    const partition = nodes.get('graphReviewList').children.at(-1);
    assert.match(partition.children[0].textContent, /по сторонам двери/);
    assert.equal(partition.children[1].children[1].textContent, 'Сплошная стена с дверью');
    await context.showResult({job_id:'reviewed', parent_job_id:'fresh-parent', saved_review:true,
        pipeline:'graph-review', stats:{}, quality:{status:'review', issues:[
            {code:'uncertain_room', message:'room 1'}, {code:'uncertain_room', message:'room 2'}]},
        warnings:['room 1','room 2','scale'], images:{graph_questions:'/questions',graph_export:'/export'},
        review_questions:[]});
    assert.equal(nodes.get('graphControls').hidden, true);
    assert.equal(nodes.get('resultTitle').textContent, 'Результат распознавания');
    assert.equal(nodes.get('windowsLabel').textContent, 'Окон');
    assert.equal(nodes.get('doorsLabel').textContent, 'Дверей');
    assert.match(nodes.get('resultMeta').textContent, /сохранённые исправления/);
    assert.match(nodes.get('roomWarningList').textContent, /room 1\nroom 2/);
    assert.match(nodes.get('recognitionWarnings').textContent, /scale/);
    assert.equal(vm.runInContext('graphSourceJob',context), 'fresh-parent');
    assert.equal(nodes.get('retrySavedReview').hidden, true);
    vm.runInContext("currentFile = {name:'focus.png'}", context);
    await context.showResult({job_id:'raw', pipeline:'v2', stats:{}, quality:{status:'invalid'},
        images:{}, review_questions:[], saved_review_available:true});
    assert.equal(nodes.get('retrySavedReview').hidden, false);
    await nodes.get('retrySavedReview').onclick();
    assert.equal(calls.at(-1).url, '/api/recognize');
    assert.deepEqual(Object.keys(calls.at(-1).data), ['file']);
    assert.doesNotMatch(html, /type="checkbox"/);
    await context.showResult({job_id:'auto-basic', pipeline:'v1', pipeline_selection:'simple_preview',
        quality:{status:'good'}, stats:{}, images:{}, review_questions:[]});
    assert.match(nodes.get('resultMeta').textContent, /Автовыбор: базовое распознавание/);
    await context.upload({name:'plan.png'}, {answers:[{choice:'door'}]});
    assert.equal(calls.at(-1).data.pipeline_v2, 'false');
    assert.equal(calls.at(-1).data.light_fill_cleanup, 'false');
    console.log('Graph UI: explicit scale, source-pixel scope and review answers passed');
})().catch(error => { console.error(error); process.exitCode = 1; });
