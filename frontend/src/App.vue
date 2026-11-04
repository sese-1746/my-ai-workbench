<script setup>
import { computed, h, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'

/* ------------------------------------------------------------------ *
 * 常量与工具
 * ------------------------------------------------------------------ */
const MB = 1024 * 1024

const PAGES = [
  { id: 'overview', label: '总览', title: '总览', desc: '服务健康、各业务状态与模型数量一览。' },
  { id: 'house', label: '房价预测', title: '房价预测', desc: '上传房价 CSV 训练 MLP 回归，再用表单估价。' },
  { id: 'sentiment', label: '情感分析', title: '情感分析', desc: '中文评论分类，BiGRU 与 TextCNN 可切换。' },
  { id: 'generation', label: '文本生成', title: '文本生成', desc: '字符级 Transformer，按开头续写文本。' },
  { id: 'qa', label: '问答检索', title: '问答检索', desc: 'TF-IDF 双通道检索，按相似度返回最相关答案。' },
  { id: 'deploy', label: '模型与部署', title: '模型与部署', desc: '运行参数、接口清单与部署方式。' },
]

const formatBytes = (value) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  if (value >= MB) return `${(value / MB).toFixed(1)} MB`
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KB`
  return `${value} B`
}
const formatDate = (iso) => {
  if (!iso) return '—'
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString('zh-CN', { hour12: false })
}
const formatPercent = (value) => (typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : '—')
const formatNumber = (value) => (typeof value === 'number' ? value.toLocaleString('zh-CN') : '—')

/* ------------------------------------------------------------------ *
 * Hash 路由
 * ------------------------------------------------------------------ */
function readHash() {
  const raw = decodeURIComponent(window.location.hash.replace(/^#\/?/, ''))
  return PAGES.some((item) => item.id === raw) ? raw : 'overview'
}
const pageId = ref(readHash())
const currentPage = computed(() => PAGES.find((item) => item.id === pageId.value) || PAGES[0])
function syncHash() {
  pageId.value = readHash()
}
function navigate(id) {
  if (pageId.value === id) return
  window.location.hash = `#/${id}`
}
watch(pageId, () => window.scrollTo({ top: 0, behavior: 'smooth' }))

/* ------------------------------------------------------------------ *
 * 请求封装：12s 普通请求 / 120s 训练与推理
 * ------------------------------------------------------------------ */
const banner = ref('')
const LONG_REQUEST = /\/(train|predict|classify|generate|query)$/

async function api(path, options = {}) {
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), LONG_REQUEST.test(path) ? 120000 : 12000)
  try {
    const response = await fetch(path, { ...options, signal: controller.signal })
    let payload = null
    try {
      payload = await response.json()
    } catch {
      payload = null
    }
    if (!response.ok) {
      const detail = payload && typeof payload.detail === 'string' ? payload.detail : `请求失败（HTTP ${response.status}）`
      throw new Error(detail)
    }
    return payload
  } catch (error) {
    let message = error.message || '请求失败'
    if (error.name === 'AbortError') message = '请求超时，请确认后端服务（端口 8000）正在运行。'
    else if (error instanceof TypeError) message = '无法连接后端服务，请确认 uvicorn 已启动。'
    banner.value = message
    throw new Error(message)
  } finally {
    clearTimeout(timer)
  }
}

/* ------------------------------------------------------------------ *
 * 通用模块工作区：模型列表 / 训练任务 / 删除
 * ------------------------------------------------------------------ */
function createModule(key) {
  const models = ref([])
  const selectedId = ref('')
  const job = ref(null)
  const busy = ref(false)
  const loading = ref(false)
  const loaded = ref(false)
  const error = ref('')
  const selected = computed(() => models.value.find((item) => item.id === selectedId.value) || null)
  let timer = null

  async function refresh() {
    loading.value = true
    try {
      const payload = await api(`/api/v1/${key}/models`)
      models.value = payload.data
      if (!models.value.some((item) => item.id === selectedId.value)) {
        selectedId.value = models.value[0]?.id || ''
      }
      loaded.value = true
      error.value = ''
    } catch (cause) {
      error.value = cause.message
    } finally {
      loading.value = false
    }
  }

  function stopPolling() {
    if (timer) {
      clearTimeout(timer)
      timer = null
    }
  }

  function poll(jobId) {
    stopPolling()
    timer = setTimeout(async () => {
      try {
        const current = await api(`/api/v1/${key}/jobs/${jobId}`)
        job.value = current
        if (current.status === 'completed') {
          busy.value = false
          await refresh()
          if (current.model?.id) selectedId.value = current.model.id
        } else if (current.status === 'failed') {
          busy.value = false
          error.value = current.error || '训练失败'
        } else {
          poll(jobId)
        }
      } catch (cause) {
        busy.value = false
        error.value = cause.message
      }
    }, 1000)
  }

  async function train(formData) {
    busy.value = true
    error.value = ''
    job.value = null
    try {
      const created = await api(`/api/v1/${key}/train`, { method: 'POST', body: formData })
      job.value = created
      poll(created.id)
    } catch (cause) {
      busy.value = false
      error.value = cause.message
    }
  }

  async function remove() {
    if (!selected.value) return
    if (!window.confirm(`确定删除模型「${selected.value.name}」？模型文件会被直接删除，无法恢复。`)) return
    error.value = ''
    try {
      await api(`/api/v1/${key}/models/${selected.value.id}`, { method: 'DELETE' })
      job.value = null
      await refresh()
    } catch (cause) {
      error.value = cause.message
    }
  }

  return { models, selectedId, selected, job, busy, loading, loaded, error, refresh, train, remove, stopPolling }
}

const progressOf = (job) => {
  if (!job || !job.epochs) return 0
  if (job.status === 'completed') return 100
  return Math.max(0, Math.min(100, Math.round(((job.current_epoch || 0) / job.epochs) * 100)))
}

/* ------------------------------------------------------------------ *
 * 内联 SVG 折线图（无图表库）
 * ------------------------------------------------------------------ */
const MetricChart = (props) => {
  const width = props.width || 640
  const height = props.height || 210
  const left = 54
  const right = 18
  const top = 32
  const bottom = 30
  const series = (props.series || [])
    .map((item) => ({ ...item, values: (item.values || []).filter((value) => typeof value === 'number' && Number.isFinite(value)) }))
    .filter((item) => item.values.length > 0)
  if (!series.length) {
    return h('p', { class: 'state empty' }, '训练开始后，这里会显示逐轮指标曲线。')
  }
  const count = Math.max(...series.map((item) => item.values.length))
  const all = series.flatMap((item) => item.values)
  const min = Math.min(...all)
  const max = Math.max(...all)
  // 自适应纵轴：指标均为非负数，留 12% 余量，避免曲线贴边或压成一条直线
  const span = max - min
  const padding = span === 0 ? Math.max(Math.abs(max) * 0.1, 0.1) : span * 0.12
  const lower = Math.max(0, min - padding)
  const upper = max + padding
  const x = (index) => (count === 1 ? left + (width - left - right) / 2 : left + (index * (width - left - right)) / (count - 1))
  const y = (value) => top + ((upper - value) / (upper - lower || 1)) * (height - top - bottom)
  const children = []
  ;[0, 0.5, 1].forEach((ratio) => {
    const tick = lower + (upper - lower) * ratio
    children.push(h('line', { x1: left, x2: width - right, y1: y(tick), y2: y(tick), stroke: '#e5e7eb', 'stroke-width': 1 }))
    children.push(h('text', { x: left - 8, y: y(tick) + 3, 'text-anchor': 'end', class: 'chart-tick' }, tick.toFixed(2)))
  })
  children.push(h('text', { x: left, y: height - 8, class: 'chart-tick' }, '第 1 轮'))
  if (count > 1) {
    children.push(h('text', { x: width - right, y: height - 8, 'text-anchor': 'end', class: 'chart-tick' }, `第 ${count} 轮`))
  }
  series.forEach((item, seriesIndex) => {
    const points = item.values.map((value, index) => `${x(index).toFixed(1)},${y(value).toFixed(1)}`).join(' ')
    children.push(
      h('polyline', {
        points,
        fill: 'none',
        stroke: item.color,
        'stroke-width': 2.2,
        'stroke-linejoin': 'round',
        'stroke-linecap': 'round',
      }),
    )
    const legendX = left + seriesIndex * 150
    children.push(h('circle', { cx: legendX, cy: 14, r: 4, fill: item.color }))
    children.push(h('text', { x: legendX + 8, y: 18, class: 'chart-legend' }, item.name))
  })
  return h('svg', { viewBox: `0 0 ${width} ${height}`, class: 'chart', role: 'img', 'aria-label': '逐轮指标曲线' }, children)
}
MetricChart.props = {
  series: { type: Array, default: () => [] },
  width: { type: Number, default: 640 },
  height: { type: Number, default: 210 },
}

/* ------------------------------------------------------------------ *
 * 总览
 * ------------------------------------------------------------------ */
const overview = reactive({ health: null, modules: [], loading: false, error: '', loaded: false })
async function loadOverview() {
  overview.loading = true
  overview.error = ''
  try {
    const [health, modules] = await Promise.all([api('/api/health'), api('/api/v1/modules')])
    overview.health = health
    overview.modules = modules.data
    overview.loaded = true
  } catch (cause) {
    overview.error = cause.message
    overview.health = null
    overview.modules = []
  } finally {
    overview.loading = false
  }
}

/* ------------------------------------------------------------------ *
 * 模块A 房价预测
 * ------------------------------------------------------------------ */
const house = createModule('house')
const houseName = ref('')
const houseEpochs = ref(60)
const houseFile = ref(null)
const houseFeatures = ref({})
const houseResult = ref(null)
const houseInferring = ref(false)

function onHouseFile(event) {
  house.error.value = ''
  houseFile.value = null
  const file = event.target.files?.[0]
  if (!file) return
  if (!file.name.toLowerCase().endsWith('.csv')) {
    house.error.value = '仅支持 UTF-8 编码的 CSV 文件'
    event.target.value = ''
    return
  }
  if (file.size > 10 * MB) {
    house.error.value = 'CSV 不能超过 10 MB'
    event.target.value = ''
    return
  }
  houseFile.value = file
}

function submitHouse() {
  house.error.value = ''
  if (!houseFile.value) {
    house.error.value = '请选择 CSV 文件'
    return
  }
  if (!houseName.value.trim()) {
    house.error.value = '请填写模型名称'
    return
  }
  if (!(houseEpochs.value >= 1 && houseEpochs.value <= 200)) {
    house.error.value = '训练轮数需在 1–200 之间'
    return
  }
  const form = new FormData()
  form.append('file', houseFile.value)
  form.append('name', houseName.value.trim())
  form.append('epochs', String(houseEpochs.value))
  house.train(form)
}

async function predictHouse() {
  if (!house.selected.value) return
  const features = {}
  Object.entries(houseFeatures.value).forEach(([key, value]) => {
    if (value !== '' && value !== null && value !== undefined) features[key] = value
  })
  house.error.value = ''
  houseInferring.value = true
  try {
    houseResult.value = await api('/api/v1/house/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: house.selected.value.id, features }),
    })
  } catch (cause) {
    house.error.value = cause.message
    houseResult.value = null
  } finally {
    houseInferring.value = false
  }
}

watch(house.selected, (model) => {
  houseResult.value = null
  const next = {}
  if (model) {
    ;(model.numeric_columns || []).forEach((column) => { next[column] = '' })
    ;(model.categorical_columns || []).forEach((column) => { next[column] = '' })
  }
  houseFeatures.value = next
})

const houseHistory = computed(() => house.job.value?.history?.length ? house.job.value.history : house.selected.value?.history || [])
const houseSeries = computed(() => [
  { name: '验证 Log RMSE', color: '#c2410c', values: houseHistory.value.map((item) => item.validation_log_rmse) },
  { name: '训练 Log RMSE', color: '#2f6feb', values: houseHistory.value.map((item) => item.train_log_rmse) },
])

/* ------------------------------------------------------------------ *
 * 模块B 情感分析
 * ------------------------------------------------------------------ */
const sentiment = createModule('sentiment')
const sentimentName = ref('')
const sentimentEpochs = ref(8)
const sentimentSource = ref('file')
const sentimentFile = ref(null)
const sentimentText = ref('')
const sentimentAlgorithm = ref('textcnn')
const sentimentTokenizer = ref('jieba')
const sentimentInput = ref('')
const sentimentResult = ref(null)
const sentimentInferring = ref(false)

function onSentimentFile(event) {
  sentiment.error.value = ''
  sentimentFile.value = null
  const file = event.target.files?.[0]
  if (!file) return
  if (!/\.(csv|jsonl)$/i.test(file.name)) {
    sentiment.error.value = '仅支持 CSV / JSONL 文件'
    event.target.value = ''
    return
  }
  if (file.size > 5 * MB) {
    sentiment.error.value = '数据文件不能超过 5 MB'
    event.target.value = ''
    return
  }
  sentimentFile.value = file
}

function submitSentiment() {
  sentiment.error.value = ''
  if (sentimentSource.value === 'file' && !sentimentFile.value) {
    sentiment.error.value = '请选择 CSV 或 JSONL 文件'
    return
  }
  if (sentimentSource.value === 'text' && !sentimentText.value.trim()) {
    sentiment.error.value = '请粘贴 CSV 数据'
    return
  }
  if (!sentimentName.value.trim()) {
    sentiment.error.value = '请填写模型名称'
    return
  }
  if (!(sentimentEpochs.value >= 1 && sentimentEpochs.value <= 20)) {
    sentiment.error.value = '训练轮数需在 1–20 之间'
    return
  }
  const form = new FormData()
  if (sentimentSource.value === 'file') form.append('file', sentimentFile.value)
  else form.append('text', sentimentText.value)
  form.append('name', sentimentName.value.trim())
  form.append('algorithm', sentimentAlgorithm.value)
  form.append('tokenizer', sentimentTokenizer.value)
  form.append('epochs', String(sentimentEpochs.value))
  sentiment.train(form)
}

async function classifySentiment() {
  if (!sentiment.selected.value) return
  if (!sentimentInput.value.trim()) {
    sentiment.error.value = '请输入需要分析的中文文本'
    return
  }
  if (sentimentInput.value.length > 2000) {
    sentiment.error.value = '文本不能超过 2000 字'
    return
  }
  sentiment.error.value = ''
  sentimentInferring.value = true
  try {
    sentimentResult.value = await api('/api/v1/sentiment/classify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: sentiment.selected.value.id, text: sentimentInput.value }),
    })
  } catch (cause) {
    sentiment.error.value = cause.message
    sentimentResult.value = null
  } finally {
    sentimentInferring.value = false
  }
}

const sentimentHistory = computed(() => sentiment.job.value?.history?.length ? sentiment.job.value.history : sentiment.selected.value?.history || [])
const sentimentSeries = computed(() => [
  { name: '验证准确率', color: '#2f6feb', values: sentimentHistory.value.map((item) => item.validation_accuracy) },
  { name: '训练准确率', color: '#60a5fa', values: sentimentHistory.value.map((item) => item.train_accuracy) },
])
const sentimentLossSeries = computed(() => [
  { name: '验证 Loss', color: '#c2410c', values: sentimentHistory.value.map((item) => item.validation_loss) },
  { name: '训练 Loss', color: '#f59e0b', values: sentimentHistory.value.map((item) => item.train_loss) },
])

/* ------------------------------------------------------------------ *
 * 模块C 文本生成
 * ------------------------------------------------------------------ */
const generation = createModule('generation')
const generationName = ref('')
const generationEpochs = ref(5)
const generationSource = ref('file')
const generationFile = ref(null)
const generationText = ref('')
const generationPrompt = ref('')
const generationLength = ref(60)
const generationTemperature = ref(0.8)
const generationTopK = ref(20)
const generationResult = ref(null)
const generationInferring = ref(false)

function onGenerationFile(event) {
  generation.error.value = ''
  generationFile.value = null
  const file = event.target.files?.[0]
  if (!file) return
  if (!/\.(txt|csv)$/i.test(file.name)) {
    generation.error.value = '仅支持 TXT / CSV 文件'
    event.target.value = ''
    return
  }
  if (file.size > 2 * MB) {
    generation.error.value = '语料文件不能超过 2 MB'
    event.target.value = ''
    return
  }
  generationFile.value = file
}

function submitGeneration() {
  generation.error.value = ''
  if (generationSource.value === 'file' && !generationFile.value) {
    generation.error.value = '请选择 TXT 或 CSV 文件'
    return
  }
  if (generationSource.value === 'text' && !generationText.value.trim()) {
    generation.error.value = '请粘贴训练语料'
    return
  }
  if (!generationName.value.trim()) {
    generation.error.value = '请填写模型名称'
    return
  }
  if (!(generationEpochs.value >= 1 && generationEpochs.value <= 20)) {
    generation.error.value = '训练轮数需在 1–20 之间'
    return
  }
  const form = new FormData()
  if (generationSource.value === 'file') form.append('file', generationFile.value)
  else form.append('text', generationText.value)
  form.append('name', generationName.value.trim())
  form.append('epochs', String(generationEpochs.value))
  generation.train(form)
}

async function generateText() {
  if (!generation.selected.value) return
  if (!generationPrompt.value.trim()) {
    generation.error.value = '请填写开头文本'
    return
  }
  if (generationPrompt.value.length > 200) {
    generation.error.value = '开头不能超过 200 字'
    return
  }
  if (!(generationLength.value >= 10 && generationLength.value <= 120)) {
    generation.error.value = '生成长度需在 10–120 之间'
    return
  }
  if (!(generationTemperature.value >= 0.2 && generationTemperature.value <= 1.5)) {
    generation.error.value = '温度需在 0.2–1.5 之间'
    return
  }
  if (!(generationTopK.value >= 1 && generationTopK.value <= 50)) {
    generation.error.value = 'top-k 需在 1–50 之间'
    return
  }
  generation.error.value = ''
  generationInferring.value = true
  try {
    generationResult.value = await api('/api/v1/generation/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model_id: generation.selected.value.id,
        prompt: generationPrompt.value,
        max_new_tokens: generationLength.value,
        temperature: generationTemperature.value,
        top_k: generationTopK.value,
      }),
    })
  } catch (cause) {
    generation.error.value = cause.message
    generationResult.value = null
  } finally {
    generationInferring.value = false
  }
}

const generationHistory = computed(() => generation.job.value?.history?.length ? generation.job.value.history : generation.selected.value?.history || [])
const generationSeries = computed(() => [
  { name: '验证 Loss', color: '#c2410c', values: generationHistory.value.map((item) => item.validation_loss) },
  { name: '训练 Loss', color: '#2f6feb', values: generationHistory.value.map((item) => item.train_loss) },
])

/* ------------------------------------------------------------------ *
 * 模块D 问答检索
 * ------------------------------------------------------------------ */
const qa = createModule('qa')
const qaName = ref('')
const qaEpochs = ref(4)
const qaSource = ref('file')
const qaFile = ref(null)
const qaText = ref('')
const qaQuestion = ref('')
const qaTopK = ref(3)
const qaResult = ref(null)
const qaInferring = ref(false)

function onQaFile(event) {
  qa.error.value = ''
  qaFile.value = null
  const file = event.target.files?.[0]
  if (!file) return
  if (!/\.(csv|jsonl)$/i.test(file.name)) {
    qa.error.value = '仅支持 CSV / JSONL 文件'
    event.target.value = ''
    return
  }
  if (file.size > 5 * MB) {
    qa.error.value = '问答文件不能超过 5 MB'
    event.target.value = ''
    return
  }
  qaFile.value = file
}

function submitQa() {
  qa.error.value = ''
  if (qaSource.value === 'file' && !qaFile.value) {
    qa.error.value = '请选择 CSV 或 JSONL 文件'
    return
  }
  if (qaSource.value === 'text' && !qaText.value.trim()) {
    qa.error.value = '请粘贴问答数据'
    return
  }
  if (!qaName.value.trim()) {
    qa.error.value = '请填写模型名称'
    return
  }
  if (!(qaEpochs.value >= 1 && qaEpochs.value <= 8)) {
    qa.error.value = '配置搜索轮数需在 1–8 之间'
    return
  }
  const form = new FormData()
  if (qaSource.value === 'file') form.append('file', qaFile.value)
  else form.append('text', qaText.value)
  form.append('name', qaName.value.trim())
  form.append('epochs', String(qaEpochs.value))
  qa.train(form)
}

async function queryQa() {
  if (!qa.selected.value) return
  if (!qaQuestion.value.trim()) {
    qa.error.value = '请输入问题'
    return
  }
  if (qaQuestion.value.length > 300) {
    qa.error.value = '问题不能超过 300 字'
    return
  }
  if (!(qaTopK.value >= 1 && qaTopK.value <= 10)) {
    qa.error.value = 'top-k 需在 1–10 之间'
    return
  }
  qa.error.value = ''
  qaInferring.value = true
  try {
    qaResult.value = await api('/api/v1/qa/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_id: qa.selected.value.id, question: qaQuestion.value, top_k: qaTopK.value }),
    })
  } catch (cause) {
    qa.error.value = cause.message
    qaResult.value = null
  } finally {
    qaInferring.value = false
  }
}

const qaHistory = computed(() => qa.job.value?.history?.length ? qa.job.value.history : qa.selected.value?.history || [])
const qaSeries = computed(() => [
  { name: '验证 MRR', color: '#2f6feb', values: qaHistory.value.map((item) => item.validation_mrr) },
  { name: '验证 Recall@1', color: '#0f766e', values: qaHistory.value.map((item) => item.validation_recall_at_1) },
])

/* ------------------------------------------------------------------ *
 * 生命周期
 * ------------------------------------------------------------------ */
function refreshAll() {
  loadOverview()
  house.refresh()
  sentiment.refresh()
  generation.refresh()
  qa.refresh()
}

onMounted(() => {
  window.addEventListener('hashchange', syncHash)
  refreshAll()
})

onBeforeUnmount(() => {
  window.removeEventListener('hashchange', syncHash)
  ;[house, sentiment, generation, qa].forEach((module) => module.stopPolling())
})
</script>

<template>
  <div class="shell">
    <aside class="sidebar">
      <div class="brand">
        <span class="logo">AI</span>
        <div>
          <strong>我的 AI 工作台</strong>
          <small>本地训练 · CPU 可跑</small>
        </div>
      </div>
      <nav class="nav">
        <button
          v-for="item in PAGES"
          :key="item.id"
          :class="['nav-item', { active: pageId === item.id }]"
          @click="navigate(item.id)"
        >
          {{ item.label }}
        </button>
      </nav>
      <footer class="side-foot">
        <span>FastAPI · PyTorch CPU</span>
        <span>模型仅存本机</span>
      </footer>
    </aside>

    <main class="main">
      <header class="page-head">
        <div>
          <h1>{{ currentPage.title }}</h1>
          <p>{{ currentPage.desc }}</p>
        </div>
        <button class="ghost" @click="refreshAll">刷新全部</button>
      </header>

      <div v-if="banner" class="alert global-alert">
        <span>{{ banner }}</span>
        <button class="link" @click="banner = ''">关闭</button>
      </div>

      <!-- ============ 总览 ============ -->
      <section v-if="pageId === 'overview'" class="stack">
        <div v-if="overview.loading" class="card state">正在检查服务状态…</div>
        <div v-else-if="overview.error" class="card state error">
          <p>{{ overview.error }}</p>
          <button class="ghost" @click="loadOverview">重试</button>
        </div>
        <template v-else>
          <div class="grid two">
            <article class="card">
              <header class="card-head">
                <h2>服务健康</h2>
                <span :class="['pill', overview.health?.status === 'ok' ? 'ok' : 'warn']">
                  {{ overview.health?.status === 'ok' ? '运行中' : '未知' }}
                </span>
              </header>
              <dl class="meta-grid">
                <div><dt>服务</dt><dd>{{ overview.health?.service || '—' }}</dd></div>
                <div><dt>版本</dt><dd>{{ overview.health?.version || '—' }}</dd></div>
                <div><dt>已保存模型</dt><dd>{{ Object.values(overview.health?.models || {}).reduce((a, b) => a + b, 0) }} 个</dd></div>
                <div><dt>活跃训练</dt><dd>{{ Object.keys(overview.health?.training || {}).length }} 个</dd></div>
              </dl>
            </article>
            <article class="card">
              <header class="card-head"><h2>约束与说明</h2></header>
              <ul class="plain">
                <li>所有训练与推理都在本机 CPU 上完成，不调用外部 API。</li>
                <li>训练进度仅存内存，后端重启后任务记录会丢失，已保存模型不受影响。</li>
                <li>每个业务同一时间只允许一个训练任务。</li>
                <li>模型目录名是 UUID，写入完成才会出现在列表中。</li>
              </ul>
            </article>
          </div>

          <div class="grid two">
            <article v-for="item in overview.modules" :key="item.id" class="card module-card">
              <header class="card-head">
                <h2>{{ item.name }}</h2>
                <span class="pill ok">{{ item.status === 'ready' ? '可用' : item.status }}</span>
              </header>
              <p class="muted">{{ item.description }}</p>
              <div class="module-foot">
                <span>已保存 {{ item.model_count }} 个模型</span>
                <button class="link" @click="navigate(item.id)">进入 →</button>
              </div>
            </article>
          </div>
          <p v-if="!overview.modules.length" class="card state empty">暂无可用业务模块。</p>
        </template>
      </section>

      <!-- ============ 房价预测 ============ -->
      <section v-if="pageId === 'house'" class="stack">
        <div class="grid two">
          <article class="card">
            <header class="card-head"><h2>训练新模型</h2></header>
            <p class="hint">CSV 需含目标列 SalePrice，100–5000 行、不超过 100 列、10 MB 以内。</p>
            <label class="field">训练数据
              <input type="file" accept=".csv" @change="onHouseFile" />
            </label>
            <p class="muted small">{{ houseFile ? `${houseFile.name} · ${formatBytes(houseFile.size)}` : '尚未选择文件' }}</p>
            <div class="row">
              <label class="field grow">模型名称
                <input v-model="houseName" maxlength="80" placeholder="例如：Ames 房价基线" />
              </label>
              <label class="field narrow">训练轮数
                <input v-model.number="houseEpochs" type="number" min="1" max="200" />
              </label>
            </div>
            <button class="primary" :disabled="house.busy.value" @click="submitHouse">
              {{ house.busy.value ? '训练中…' : '开始训练' }}
            </button>
            <p v-if="house.error.value" class="state error inline">{{ house.error.value }}</p>
          </article>

          <article class="card">
            <header class="card-head">
              <h2>训练任务</h2>
              <span v-if="house.job.value" class="pill">{{ house.job.value.status }}</span>
            </header>
            <p v-if="!house.job.value" class="state empty">提交训练后，这里显示任务状态与逐轮进度。</p>
            <template v-else>
              <div class="progress">
                <div class="bar" :style="{ width: `${progressOf(house.job.value)}%` }"></div>
              </div>
              <p class="muted small">
                {{ house.job.value.stage }} · 第 {{ house.job.value.current_epoch }}/{{ house.job.value.epochs }} 轮
              </p>
              <p v-if="house.job.value.error" class="state error inline">{{ house.job.value.error }}</p>
            </template>
          </article>
        </div>

        <article class="card">
          <header class="card-head"><h2>逐轮曲线</h2><span class="muted small">Log RMSE 越低越好</span></header>
          <MetricChart :series="houseSeries" />
        </article>

        <article class="card">
          <header class="card-head">
            <h2>模型管理</h2>
            <button class="ghost" :disabled="house.loading.value" @click="house.refresh">刷新</button>
          </header>
          <p v-if="house.loading.value" class="state">加载模型列表…</p>
          <p v-else-if="!house.models.value.length" class="state empty">还没有模型，先上传数据训练一个吧。</p>
          <template v-else>
            <div class="row">
              <label class="field grow">当前模型
                <select v-model="house.selectedId.value">
                  <option v-for="model in house.models.value" :key="model.id" :value="model.id">
                    {{ model.name }} · {{ model.algorithm }} · {{ formatDate(model.created_at) }}
                  </option>
                </select>
              </label>
              <button class="danger" @click="house.remove">删除模型</button>
            </div>
            <div v-if="house.selected.value" class="meta-grid">
              <div><dt>测试 Log RMSE</dt><dd>{{ house.selected.value.test.log_rmse }}</dd></div>
              <div><dt>测试 MAE</dt><dd>{{ house.selected.value.test.mae }}</dd></div>
              <div><dt>测试 R²</dt><dd>{{ house.selected.value.test.r2 }}</dd></div>
              <div><dt>最佳轮次</dt><dd>{{ house.selected.value.best_epoch }} / {{ house.selected.value.epochs }}</dd></div>
              <div><dt>参数量</dt><dd>{{ formatNumber(house.selected.value.param_count) }}</dd></div>
              <div><dt>训练耗时</dt><dd>{{ house.selected.value.train_seconds }} s</dd></div>
              <div><dt>模型体积</dt><dd>{{ formatBytes(house.selected.value.model_bytes) }}</dd></div>
              <div><dt>训练 / 验证 / 测试</dt><dd>{{ house.selected.value.train_count }} / {{ house.selected.value.validation_count }} / {{ house.selected.value.test_count }}</dd></div>
              <div><dt>数据指纹</dt><dd class="mono">{{ house.selected.value.data_fingerprint.slice(0, 16) }}…</dd></div>
              <div><dt>创建时间</dt><dd>{{ formatDate(house.selected.value.created_at) }}</dd></div>
            </div>
          </template>
        </article>

        <article class="card">
          <header class="card-head"><h2>房屋估价</h2><span class="muted small">未填特征使用训练集默认值</span></header>
          <p v-if="!house.selected.value" class="state empty">请先训练或选择一个模型。</p>
          <template v-else>
            <div class="fields">
              <label v-for="column in house.selected.value.numeric_columns" :key="column" class="field">
                {{ column }}
                <input v-model="houseFeatures[column]" type="number" step="any" :placeholder="String(house.selected.value.defaults[column])" />
              </label>
              <label v-for="column in house.selected.value.categorical_columns" :key="column" class="field">
                {{ column }}
                <select v-model="houseFeatures[column]">
                  <option value="">默认：{{ house.selected.value.defaults[column] }}</option>
                  <option v-for="value in house.selected.value.categories[column]" :key="value" :value="value">{{ value }}</option>
                </select>
              </label>
            </div>
            <button class="primary" :disabled="houseInferring" @click="predictHouse">
              {{ houseInferring ? '计算中…' : '估算价格' }}
            </button>
            <div v-if="houseResult" class="result">
              <strong>预估价格：${{ houseResult.data.price.toLocaleString('zh-CN') }}</strong>
              <p class="muted small">{{ houseResult.notice }}</p>
            </div>
          </template>
        </article>
      </section>

      <!-- ============ 情感分析 ============ -->
      <section v-if="pageId === 'sentiment'" class="stack">
        <div class="grid two">
          <article class="card">
            <header class="card-head"><h2>训练新模型</h2></header>
            <p class="hint">CSV / JSONL，字段 text + label（也接受 正文 + 类别），50–5000 条、2–12 类、每类 ≥5 条、5 MB 以内。</p>
            <div class="tabs">
              <button :class="['tab', { active: sentimentSource === 'file' }]" @click="sentimentSource = 'file'">上传文件</button>
              <button :class="['tab', { active: sentimentSource === 'text' }]" @click="sentimentSource = 'text'">粘贴 CSV</button>
            </div>
            <label v-if="sentimentSource === 'file'" class="field">训练数据
              <input type="file" accept=".csv,.jsonl" @change="onSentimentFile" />
            </label>
            <p v-if="sentimentSource === 'file'" class="muted small">
              {{ sentimentFile ? `${sentimentFile.name} · ${formatBytes(sentimentFile.size)}` : '尚未选择文件' }}
            </p>
            <label v-else class="field">CSV 文本
              <textarea v-model="sentimentText" rows="6" placeholder="text,label&#10;质量很好,正面"></textarea>
            </label>
            <div class="row">
              <label class="field grow">模型名称
                <input v-model="sentimentName" maxlength="80" placeholder="例如：评论情感 TextCNN" />
              </label>
              <label class="field narrow">训练轮数
                <input v-model.number="sentimentEpochs" type="number" min="1" max="20" />
              </label>
            </div>
            <div class="row">
              <label class="field grow">算法
                <select v-model="sentimentAlgorithm">
                  <option value="textcnn">TextCNN（卷积核 3/4/5）</option>
                  <option value="bigru">双向 GRU（64 维）</option>
                </select>
              </label>
              <label class="field grow">分词器
                <select v-model="sentimentTokenizer">
                  <option value="jieba">jieba</option>
                  <option value="regex">regex（英文）</option>
                </select>
              </label>
            </div>
            <button class="primary" :disabled="sentiment.busy.value" @click="submitSentiment">
              {{ sentiment.busy.value ? '训练中…' : '开始训练' }}
            </button>
            <p v-if="sentiment.error.value" class="state error inline">{{ sentiment.error.value }}</p>
          </article>

          <article class="card">
            <header class="card-head">
              <h2>训练任务</h2>
              <span v-if="sentiment.job.value" class="pill">{{ sentiment.job.value.status }}</span>
            </header>
            <p v-if="!sentiment.job.value" class="state empty">提交训练后，这里显示任务状态与逐轮进度。</p>
            <template v-else>
              <div class="progress">
                <div class="bar" :style="{ width: `${progressOf(sentiment.job.value)}%` }"></div>
              </div>
              <p class="muted small">
                {{ sentiment.job.value.stage }} · 第 {{ sentiment.job.value.current_epoch }}/{{ sentiment.job.value.epochs }} 轮
              </p>
              <p v-if="sentiment.job.value.error" class="state error inline">{{ sentiment.job.value.error }}</p>
            </template>
          </article>
        </div>

        <div class="grid two">
          <article class="card">
            <header class="card-head"><h2>验证准确率</h2></header>
            <MetricChart :series="sentimentSeries" :width="360" />
          </article>
          <article class="card">
            <header class="card-head"><h2>损失曲线</h2></header>
            <MetricChart :series="sentimentLossSeries" :width="360" />
          </article>
        </div>

        <article class="card">
          <header class="card-head">
            <h2>模型管理</h2>
            <button class="ghost" :disabled="sentiment.loading.value" @click="sentiment.refresh">刷新</button>
          </header>
          <p v-if="sentiment.loading.value" class="state">加载模型列表…</p>
          <p v-else-if="!sentiment.models.value.length" class="state empty">还没有模型，先上传数据训练一个吧。</p>
          <template v-else>
            <div class="row">
              <label class="field grow">当前模型
                <select v-model="sentiment.selectedId.value">
                  <option v-for="model in sentiment.models.value" :key="model.id" :value="model.id">
                    {{ model.name }} · {{ model.algorithm_label }} · {{ formatDate(model.created_at) }}
                  </option>
                </select>
              </label>
              <button class="danger" @click="sentiment.remove">删除模型</button>
            </div>
            <div v-if="sentiment.selected.value" class="meta-grid">
              <div><dt>测试准确率</dt><dd>{{ formatPercent(sentiment.selected.value.test.accuracy) }}</dd></div>
              <div><dt>测试 Macro-F1</dt><dd>{{ sentiment.selected.value.test.macro_f1 }}</dd></div>
              <div><dt>最佳轮次</dt><dd>{{ sentiment.selected.value.best_epoch }} / {{ sentiment.selected.value.epochs }}</dd></div>
              <div><dt>词表大小</dt><dd>{{ sentiment.selected.value.vocab_size }}</dd></div>
              <div><dt>参数量</dt><dd>{{ formatNumber(sentiment.selected.value.param_count) }}</dd></div>
              <div><dt>训练耗时</dt><dd>{{ sentiment.selected.value.train_seconds }} s</dd></div>
              <div><dt>模型体积</dt><dd>{{ formatBytes(sentiment.selected.value.model_bytes) }}</dd></div>
              <div><dt>样本数</dt><dd>{{ sentiment.selected.value.row_count }}</dd></div>
            </div>
            <div v-if="sentiment.selected.value" class="matrix-wrap">
              <h3>测试集混淆矩阵</h3>
              <table class="matrix">
                <thead>
                  <tr>
                    <th>真实 \ 预测</th>
                    <th v-for="label in sentiment.selected.value.labels" :key="label">{{ label }}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(label, rowIndex) in sentiment.selected.value.labels" :key="label">
                    <th>{{ label }}</th>
                    <td v-for="(_, columnIndex) in sentiment.selected.value.labels" :key="columnIndex">
                      {{ sentiment.selected.value.test.confusion_matrix[rowIndex][columnIndex] }}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </template>
        </article>

        <article class="card">
          <header class="card-head"><h2>文本分类</h2><span class="muted small">概率未经校准</span></header>
          <p v-if="!sentiment.selected.value" class="state empty">请先训练或选择一个模型。</p>
          <template v-else>
            <label class="field">待分析文本
              <textarea v-model="sentimentInput" rows="4" maxlength="2000" placeholder="例如：物流很快，包装完好，非常满意。"></textarea>
            </label>
            <p class="muted small">{{ sentimentInput.length }} / 2000 字</p>
            <button class="primary" :disabled="sentimentInferring" @click="classifySentiment">
              {{ sentimentInferring ? '分析中…' : '开始分析' }}
            </button>
            <div v-if="sentimentResult" class="result">
              <p><strong>预测类别：{{ sentimentResult.data.label }}</strong>（置信度 {{ formatPercent(sentimentResult.data.confidence) }}）</p>
              <div class="bars">
                <div v-for="item in sentimentResult.data.probabilities" :key="item.label" class="bar-row">
                  <span class="bar-label">{{ item.label }}</span>
                  <div class="bar-track"><div class="bar-fill" :style="{ width: `${Math.max(item.probability * 100, 1)}%` }"></div></div>
                  <span class="mono">{{ formatPercent(item.probability) }}</span>
                </div>
              </div>
              <p class="muted small">分词预览：{{ sentimentResult.data.tokens.join(' / ') }}</p>
              <p class="muted small">{{ sentimentResult.notice }}</p>
            </div>
          </template>
        </article>
      </section>

      <!-- ============ 文本生成 ============ -->
      <section v-if="pageId === 'generation'" class="stack">
        <div class="grid two">
          <article class="card">
            <header class="card-head"><h2>训练新模型</h2></header>
            <p class="hint">TXT 每行一段，或 CSV 含 content/正文 列；10–2000 行、总长 ≤12 万字符、2 MB 以内。</p>
            <div class="tabs">
              <button :class="['tab', { active: generationSource === 'file' }]" @click="generationSource = 'file'">上传文件</button>
              <button :class="['tab', { active: generationSource === 'text' }]" @click="generationSource = 'text'">粘贴语料</button>
            </div>
            <label v-if="generationSource === 'file'" class="field">训练语料
              <input type="file" accept=".txt,.csv" @change="onGenerationFile" />
            </label>
            <p v-if="generationSource === 'file'" class="muted small">
              {{ generationFile ? `${generationFile.name} · ${formatBytes(generationFile.size)}` : '尚未选择文件' }}
            </p>
            <label v-else class="field">语料文本
              <textarea v-model="generationText" rows="6" placeholder="每行一段文字，至少 10 行。"></textarea>
            </label>
            <div class="row">
              <label class="field grow">模型名称
                <input v-model="generationName" maxlength="80" placeholder="例如：小语料生成器" />
              </label>
              <label class="field narrow">训练轮数
                <input v-model.number="generationEpochs" type="number" min="1" max="20" />
              </label>
            </div>
            <button class="primary" :disabled="generation.busy.value" @click="submitGeneration">
              {{ generation.busy.value ? '训练中…' : '开始训练' }}
            </button>
            <p v-if="generation.error.value" class="state error inline">{{ generation.error.value }}</p>
          </article>

          <article class="card">
            <header class="card-head">
              <h2>训练任务</h2>
              <span v-if="generation.job.value" class="pill">{{ generation.job.value.status }}</span>
            </header>
            <p v-if="!generation.job.value" class="state empty">提交训练后，这里显示任务状态与逐轮进度。</p>
            <template v-else>
              <div class="progress">
                <div class="bar" :style="{ width: `${progressOf(generation.job.value)}%` }"></div>
              </div>
              <p class="muted small">
                {{ generation.job.value.stage }} · 第 {{ generation.job.value.current_epoch }}/{{ generation.job.value.epochs }} 轮
              </p>
              <p v-if="generation.job.value.error" class="state error inline">{{ generation.job.value.error }}</p>
            </template>
          </article>
        </div>

        <article class="card">
          <header class="card-head"><h2>逐轮损失</h2><span class="muted small">字符级交叉熵</span></header>
          <MetricChart :series="generationSeries" />
        </article>

        <article class="card">
          <header class="card-head">
            <h2>模型管理</h2>
            <button class="ghost" :disabled="generation.loading.value" @click="generation.refresh">刷新</button>
          </header>
          <p v-if="generation.loading.value" class="state">加载模型列表…</p>
          <p v-else-if="!generation.models.value.length" class="state empty">还没有模型，先上传语料训练一个吧。</p>
          <template v-else>
            <div class="row">
              <label class="field grow">当前模型
                <select v-model="generation.selectedId.value">
                  <option v-for="model in generation.models.value" :key="model.id" :value="model.id">
                    {{ model.name }} · {{ model.algorithm_label }} · {{ formatDate(model.created_at) }}
                  </option>
                </select>
              </label>
              <button class="danger" @click="generation.remove">删除模型</button>
            </div>
            <div v-if="generation.selected.value" class="meta-grid">
              <div><dt>测试困惑度</dt><dd>{{ generation.selected.value.test.perplexity }}</dd></div>
              <div><dt>下一字符准确率</dt><dd>{{ formatPercent(generation.selected.value.test.next_char_accuracy) }}</dd></div>
              <div><dt>最佳轮次</dt><dd>{{ generation.selected.value.best_epoch }} / {{ generation.selected.value.epochs }}</dd></div>
              <div><dt>词表大小</dt><dd>{{ generation.selected.value.vocab_size }}</dd></div>
              <div><dt>上下文长度</dt><dd>{{ generation.selected.value.context }}</dd></div>
              <div><dt>参数量</dt><dd>{{ formatNumber(generation.selected.value.param_count) }}</dd></div>
              <div><dt>训练耗时</dt><dd>{{ generation.selected.value.train_seconds }} s</dd></div>
              <div><dt>模型体积</dt><dd>{{ formatBytes(generation.selected.value.model_bytes) }}</dd></div>
            </div>
          </template>
        </article>

        <article class="card">
          <header class="card-head"><h2>续写</h2><span class="muted small">top-k + temperature 采样</span></header>
          <p v-if="!generation.selected.value" class="state empty">请先训练或选择一个模型。</p>
          <template v-else>
            <label class="field">开头
              <textarea v-model="generationPrompt" rows="3" maxlength="200" placeholder="例如：人工智能正在"></textarea>
            </label>
            <div class="row">
              <label class="field grow">生成长度（10–120）
                <input v-model.number="generationLength" type="number" min="10" max="120" />
              </label>
              <label class="field grow">温度（0.2–1.5）
                <input v-model.number="generationTemperature" type="number" min="0.2" max="1.5" step="0.1" />
              </label>
              <label class="field grow">top-k（1–50）
                <input v-model.number="generationTopK" type="number" min="1" max="50" />
              </label>
            </div>
            <button class="primary" :disabled="generationInferring" @click="generateText">
              {{ generationInferring ? '生成中…' : '生成续写' }}
            </button>
            <div v-if="generationResult" class="result">
              <p class="generated">{{ generationResult.data.full_text }}</p>
              <p class="muted small">
                本次生成 {{ generationResult.data.generated_chars }} 字 ·
                {{ generationResult.data.stopped_by_eos ? '遇到结束符提前停止' : '达到设定长度' }}
              </p>
              <p class="muted small">{{ generationResult.notice }}</p>
            </div>
          </template>
        </article>
      </section>

      <!-- ============ 问答检索 ============ -->
      <section v-if="pageId === 'qa'" class="stack">
        <div class="grid two">
          <article class="card">
            <header class="card-head"><h2>训练新模型</h2></header>
            <p class="hint">CSV / JSONL，字段 question + answer；10–2000 组、5 MB 以内。每轮搜索一组检索配置。</p>
            <div class="tabs">
              <button :class="['tab', { active: qaSource === 'file' }]" @click="qaSource = 'file'">上传文件</button>
              <button :class="['tab', { active: qaSource === 'text' }]" @click="qaSource = 'text'">粘贴 CSV</button>
            </div>
            <label v-if="qaSource === 'file'" class="field">问答数据
              <input type="file" accept=".csv,.jsonl" @change="onQaFile" />
            </label>
            <p v-if="qaSource === 'file'" class="muted small">
              {{ qaFile ? `${qaFile.name} · ${formatBytes(qaFile.size)}` : '尚未选择文件' }}
            </p>
            <label v-else class="field">CSV 文本
              <textarea v-model="qaText" rows="6" placeholder="question,answer&#10;如何重置路由器,长按重置孔 10 秒"></textarea>
            </label>
            <div class="row">
              <label class="field grow">模型名称
                <input v-model="qaName" maxlength="80" placeholder="例如：客服问答索引" />
              </label>
              <label class="field narrow">搜索轮数
                <input v-model.number="qaEpochs" type="number" min="1" max="8" />
              </label>
            </div>
            <button class="primary" :disabled="qa.busy.value" @click="submitQa">
              {{ qa.busy.value ? '训练中…' : '开始训练' }}
            </button>
            <p v-if="qa.error.value" class="state error inline">{{ qa.error.value }}</p>
          </article>

          <article class="card">
            <header class="card-head">
              <h2>训练任务</h2>
              <span v-if="qa.job.value" class="pill">{{ qa.job.value.status }}</span>
            </header>
            <p v-if="!qa.job.value" class="state empty">提交训练后，这里显示任务状态与逐轮进度。</p>
            <template v-else>
              <div class="progress">
                <div class="bar" :style="{ width: `${progressOf(qa.job.value)}%` }"></div>
              </div>
              <p class="muted small">
                {{ qa.job.value.stage }} · 第 {{ qa.job.value.current_epoch }}/{{ qa.job.value.epochs }} 轮
              </p>
              <p v-if="qa.job.value.error" class="state error inline">{{ qa.job.value.error }}</p>
            </template>
          </article>
        </div>

        <article class="card">
          <header class="card-head"><h2>验证指标</h2><span class="muted small">MRR 与 Recall@1 越高越好</span></header>
          <MetricChart :series="qaSeries" />
        </article>

        <article class="card">
          <header class="card-head">
            <h2>模型管理</h2>
            <button class="ghost" :disabled="qa.loading.value" @click="qa.refresh">刷新</button>
          </header>
          <p v-if="qa.loading.value" class="state">加载模型列表…</p>
          <p v-else-if="!qa.models.value.length" class="state empty">还没有模型，先上传问答数据训练一个吧。</p>
          <template v-else>
            <div class="row">
              <label class="field grow">当前模型
                <select v-model="qa.selectedId.value">
                  <option v-for="model in qa.models.value" :key="model.id" :value="model.id">
                    {{ model.name }} · {{ formatDate(model.created_at) }}
                  </option>
                </select>
              </label>
              <button class="danger" @click="qa.remove">删除模型</button>
            </div>
            <div v-if="qa.selected.value" class="meta-grid">
              <div><dt>测试 Recall@1</dt><dd>{{ formatPercent(qa.selected.value.test.recall_at_1) }}</dd></div>
              <div><dt>测试 Recall@3</dt><dd>{{ formatPercent(qa.selected.value.test.recall_at_3) }}</dd></div>
              <div><dt>测试 MRR</dt><dd>{{ qa.selected.value.test.mrr }}</dd></div>
              <div><dt>最佳轮次</dt><dd>{{ qa.selected.value.best_epoch }} / {{ qa.selected.value.epochs }}</dd></div>
              <div><dt>答案条数</dt><dd>{{ qa.selected.value.answer_count }}</dd></div>
              <div><dt>训练耗时</dt><dd>{{ qa.selected.value.train_seconds }} s</dd></div>
              <div><dt>模型体积</dt><dd>{{ formatBytes(qa.selected.value.model_bytes) }}</dd></div>
              <div><dt>最优配置</dt><dd class="mono">{{ qa.selected.value.best_config.word_ngram.join('-') }} / {{ qa.selected.value.best_config.char_ngram.join('-') }} / w={{ qa.selected.value.best_config.word_weight }}</dd></div>
            </div>
          </template>
        </article>

        <article class="card">
          <header class="card-head"><h2>检索</h2><span class="muted small">按字面相似度排序</span></header>
          <p v-if="!qa.selected.value" class="state empty">请先训练或选择一个模型。</p>
          <template v-else>
            <div class="row">
              <label class="field grow">你的问题
                <input v-model="qaQuestion" maxlength="300" placeholder="例如：路由器怎么恢复出厂设置" />
              </label>
              <label class="field narrow">top-k
                <input v-model.number="qaTopK" type="number" min="1" max="10" />
              </label>
            </div>
            <button class="primary" :disabled="qaInferring" @click="queryQa">
              {{ qaInferring ? '检索中…' : '开始检索' }}
            </button>
            <div v-if="qaResult" class="result">
              <div v-for="item in qaResult.data.results" :key="item.rank" class="hit">
                <div class="hit-head">
                  <strong>#{{ item.rank }} · 相似度 {{ item.score }}</strong>
                  <span class="muted small">原问题：{{ item.question }}</span>
                </div>
                <p>{{ item.answer }}</p>
              </div>
              <p class="muted small">{{ qaResult.notice }}</p>
            </div>
          </template>
        </article>
      </section>

      <!-- ============ 模型与部署 ============ -->
      <section v-if="pageId === 'deploy'" class="stack">
        <div class="grid two">
          <article class="card">
            <header class="card-head"><h2>运行状态</h2></header>
            <dl class="meta-grid">
              <div><dt>后端</dt><dd>{{ overview.health?.status === 'ok' ? '已连接（/api/health）' : '未连接' }}</dd></div>
              <div><dt>版本</dt><dd>{{ overview.health?.version || '—' }}</dd></div>
              <div><dt>模型总数</dt><dd>{{ Object.values(overview.health?.models || {}).reduce((a, b) => a + b, 0) }}</dd></div>
              <div><dt>当前训练</dt><dd>{{ Object.keys(overview.health?.training || {}).length }}</dd></div>
            </dl>
          </article>
          <article class="card">
            <header class="card-head"><h2>启动命令</h2></header>
            <pre class="code">cd backend
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

cd frontend
npm install
npm run dev   # http://127.0.0.1:5175

# Docker
docker compose up --build</pre>
          </article>
        </div>

        <article class="card">
          <header class="card-head"><h2>接口清单</h2></header>
          <div class="table-wrap">
            <table class="matrix wide">
              <thead><tr><th>方法</th><th>路径</th><th>说明</th></tr></thead>
              <tbody>
                <tr><td>GET</td><td class="mono">/api/health</td><td>健康检查与各模块模型数量</td></tr>
                <tr><td>GET</td><td class="mono">/api/v1/modules</td><td>业务模块清单</td></tr>
                <tr><td>POST</td><td class="mono">/api/v1/house/train</td><td>上传 CSV，返回 202 + job_id</td></tr>
                <tr><td>GET</td><td class="mono">/api/v1/house/jobs/{id}</td><td>任务状态与逐轮 history</td></tr>
                <tr><td>GET</td><td class="mono">/api/v1/house/models</td><td>模型清单（按创建时间倒序）</td></tr>
                <tr><td>DELETE</td><td class="mono">/api/v1/house/models/{id}</td><td>删除模型目录</td></tr>
                <tr><td>POST</td><td class="mono">/api/v1/house/predict</td><td>特征估价</td></tr>
                <tr><td>POST</td><td class="mono">/api/v1/sentiment/train</td><td>CSV/JSONL 或粘贴文本训练</td></tr>
                <tr><td>POST</td><td class="mono">/api/v1/sentiment/classify</td><td>类别概率 + 分词预览</td></tr>
                <tr><td>POST</td><td class="mono">/api/v1/generation/train</td><td>TXT/CSV 语料训练</td></tr>
                <tr><td>POST</td><td class="mono">/api/v1/generation/generate</td><td>top-k + 温度采样续写</td></tr>
                <tr><td>POST</td><td class="mono">/api/v1/qa/train</td><td>问答对训练（配置搜索）</td></tr>
                <tr><td>POST</td><td class="mono">/api/v1/qa/query</td><td>返回 top-k 答案</td></tr>
              </tbody>
            </table>
          </div>
          <p class="muted small">每个业务都有 jobs / models / DELETE models 三件套，上表只列了差异端点。</p>
        </article>

        <article class="card">
          <header class="card-head"><h2>限制与已知问题</h2></header>
          <ul class="plain">
            <li>训练进度保存在内存，服务重启后 jobs 丢失，前端会提示"任务不存在"。</li>
            <li>同一模块同时只允许一个训练任务，重复提交会返回 422。</li>
            <li>上传上限：房价 10 MB、情感 5 MB、生成 2 MB、问答 5 MB；扩展名不符返回 415。</li>
            <li>模型目录由环境变量覆盖：HOUSE_MODEL_DIR / SENTIMENT_MODEL_DIR / GENERATION_MODEL_DIR / QA_MODEL_DIR。</li>
            <li>artifacts/、.venv/、node_modules/ 与日志均不入库；模型权重不会提交到 Git。</li>
          </ul>
        </article>
      </section>

      <footer class="page-foot">所有数据在本机处理 · 模型制品不进入 Git 仓库</footer>
    </main>
  </div>
</template>

<style>
:root {
  --bg: #f5f6f8;
  --panel: #ffffff;
  --ink: #1f2430;
  --muted: #6b7280;
  --line: #e5e7eb;
  --accent: #2f6feb;
  --accent-soft: #eaf1ff;
  --ok: #137a4b;
  --ok-soft: #e8f6ee;
  --warn: #b45309;
  --warn-soft: #fdf3e3;
  --err: #b42318;
  --err-soft: #fdecea;
  --radius: 12px;
  color-scheme: light;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}

button, input, select, textarea { font: inherit; color: inherit; }

.shell { display: flex; min-height: 100vh; }

.sidebar {
  width: 224px;
  flex: 0 0 224px;
  background: #101828;
  color: #e6ebf5;
  padding: 22px 16px;
  display: flex;
  flex-direction: column;
  gap: 22px;
  position: sticky;
  top: 0;
  height: 100vh;
}

.brand { display: flex; gap: 10px; align-items: center; }
.brand strong { display: block; font-size: 15px; letter-spacing: 0.02em; }
.brand small { color: #9aa6bd; font-size: 11px; }
.logo {
  width: 36px; height: 36px; border-radius: 10px;
  background: linear-gradient(135deg, #4b8bff, #7c5cff);
  display: grid; place-items: center; font-weight: 700; font-size: 13px; color: #fff;
}

.nav { display: flex; flex-direction: column; gap: 4px; }
.nav-item {
  text-align: left;
  background: transparent;
  border: 0;
  color: #c3cbdc;
  padding: 9px 12px;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
}
.nav-item:hover { background: rgba(255, 255, 255, 0.08); color: #fff; }
.nav-item.active { background: #2f6feb; color: #fff; font-weight: 600; }

.side-foot { margin-top: auto; display: flex; flex-direction: column; gap: 4px; color: #7d879b; font-size: 11px; }

.main { flex: 1; min-width: 0; padding: 28px 32px 40px; max-width: 1280px; }

.page-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; margin-bottom: 18px; }
.page-head h1 { margin: 0 0 4px; font-size: 22px; letter-spacing: 0.01em; }
.page-head p { margin: 0; color: var(--muted); }

.stack { display: flex; flex-direction: column; gap: 16px; }
.grid { display: grid; gap: 16px; }
.grid.two { grid-template-columns: repeat(2, minmax(0, 1fr)); }

.card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 18px 20px;
}
.card-head { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 12px; }
.card-head h2 { margin: 0; font-size: 15px; }
.card-head h3 { margin: 0; font-size: 13px; }

.muted { color: var(--muted); }
.small { font-size: 12px; }
.mono { font-family: "Cascadia Mono", "Consolas", monospace; font-size: 12px; }
.hint { margin: 0 0 12px; color: var(--muted); font-size: 12px; }
.plain { margin: 0; padding-left: 18px; color: #374151; }
.plain li { margin-bottom: 6px; }

.field { display: flex; flex-direction: column; gap: 6px; font-size: 12px; color: var(--muted); margin-bottom: 12px; }
.field input, .field select, .field textarea {
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 9px 11px;
  background: #fff;
  color: var(--ink);
  font-size: 13px;
  width: 100%;
}
.field input:focus, .field select:focus, .field textarea:focus {
  outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px rgba(47, 111, 235, 0.12);
}
.field textarea { resize: vertical; font-family: inherit; }
.row { display: flex; gap: 12px; align-items: flex-end; }
.grow { flex: 1; }
.narrow { width: 132px; flex: 0 0 132px; }

button.primary, button.ghost, button.danger {
  border-radius: 8px;
  padding: 9px 16px;
  cursor: pointer;
  border: 1px solid transparent;
  transition: filter 0.15s, background 0.15s;
}
button.primary { background: var(--accent); color: #fff; font-weight: 600; }
button.primary:hover:not(:disabled) { filter: brightness(1.06); }
button.ghost { background: #fff; border-color: var(--line); color: #374151; }
button.ghost:hover:not(:disabled) { border-color: #c7ccd6; }
button.danger { background: #fff; border-color: #f0c2bc; color: var(--err); }
button.danger:hover { background: var(--err-soft); }
button:disabled { opacity: 0.55; cursor: not-allowed; }
button.link { background: none; border: 0; color: var(--accent); cursor: pointer; padding: 0; }

.tabs { display: inline-flex; background: #f1f3f6; border-radius: 8px; padding: 3px; margin-bottom: 12px; }
.tab { border: 0; background: transparent; padding: 6px 14px; border-radius: 6px; cursor: pointer; color: var(--muted); font-size: 12px; }
.tab.active { background: #fff; color: var(--ink); box-shadow: 0 1px 2px rgba(16, 24, 40, 0.08); }

.pill { background: #eef1f5; color: #475467; border-radius: 999px; padding: 2px 10px; font-size: 12px; }
.pill.ok { background: var(--ok-soft); color: var(--ok); }
.pill.warn { background: var(--warn-soft); color: var(--warn); }

.state { color: var(--muted); margin: 6px 0; }
.state.empty { padding: 18px; text-align: center; background: #fafbfc; border: 1px dashed var(--line); border-radius: 8px; }
.state.error { color: var(--err); }
.state.error.inline { background: var(--err-soft); border: 1px solid #f5c6c0; border-radius: 8px; padding: 8px 10px; font-size: 12px; }

.alert { display: flex; justify-content: space-between; align-items: center; gap: 12px; border-radius: 8px; padding: 10px 14px; margin-bottom: 14px; }
.global-alert { background: var(--err-soft); border: 1px solid #f5c6c0; color: var(--err); }

.progress { height: 8px; border-radius: 999px; background: #eef1f5; overflow: hidden; }
.progress .bar { height: 100%; background: linear-gradient(90deg, #4b8bff, #7c5cff); transition: width 0.4s ease; }

.meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px 18px; margin: 4px 0 12px; }
.meta-grid dt { font-size: 11px; color: var(--muted); margin: 0; }
.meta-grid dd { margin: 0; font-size: 13px; font-weight: 600; word-break: break-all; }

.fields { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0 14px; }

.result { margin-top: 14px; background: #f8fafc; border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
.result strong { font-size: 15px; }
.generated { white-space: pre-wrap; line-height: 1.8; margin: 4px 0 10px; }

.bars { display: flex; flex-direction: column; gap: 6px; margin: 10px 0; }
.bar-row { display: grid; grid-template-columns: 70px 1fr 64px; gap: 10px; align-items: center; font-size: 12px; }
.bar-label { color: var(--muted); }
.bar-track { height: 8px; background: #e9edf3; border-radius: 999px; overflow: hidden; }
.bar-fill { height: 100%; background: var(--accent); border-radius: 999px; }

.matrix-wrap { margin-top: 8px; }
.matrix-wrap h3 { font-size: 13px; margin: 10px 0 8px; }
.table-wrap { overflow-x: auto; }
.matrix { border-collapse: collapse; font-size: 12px; width: 100%; }
.matrix th, .matrix td { border: 1px solid var(--line); padding: 6px 8px; text-align: center; }
.matrix th { background: #f8fafc; color: #475467; font-weight: 600; }
.matrix td { background: #fff; }
.matrix.wide { min-width: 560px; }
.matrix.wide td:nth-child(3) { text-align: left; }

.hit { border-top: 1px solid var(--line); padding: 10px 0; }
.hit:first-child { border-top: 0; padding-top: 0; }
.hit-head { display: flex; flex-direction: column; gap: 2px; margin-bottom: 4px; }
.hit p { margin: 0; color: #374151; }

.chart { width: 100%; height: auto; display: block; }
.chart-tick { font-size: 10px; fill: #98a2b3; font-family: inherit; }
.chart-legend { font-size: 11px; fill: #475467; font-family: inherit; }

.code {
  margin: 0;
  background: #0f172a;
  color: #e2e8f0;
  padding: 14px 16px;
  border-radius: 10px;
  font-family: "Cascadia Mono", "Consolas", monospace;
  font-size: 12px;
  line-height: 1.7;
  overflow-x: auto;
}

.module-card { display: flex; flex-direction: column; gap: 8px; }
.module-foot { margin-top: auto; display: flex; justify-content: space-between; align-items: center; padding-top: 8px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12px; }

.page-foot { margin-top: 26px; text-align: center; color: #98a2b3; font-size: 12px; }

@media (max-width: 980px) {
  .shell { flex-direction: column; }
  .sidebar { position: static; width: 100%; height: auto; flex-direction: column; padding: 14px 16px; }
  .nav { flex-direction: row; overflow-x: auto; }
  .side-foot { display: none; }
  .main { padding: 20px 16px 32px; }
  .grid.two { grid-template-columns: 1fr; }
  .row { flex-wrap: wrap; }
  .narrow { width: 100%; flex: 1 1 120px; }
}
</style>
