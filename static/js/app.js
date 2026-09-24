/**
 * Vanilla JavaScript Frontend Engine for Ujjain Onion Price Prediction System
 * Communicates directly with FastAPI backend via REST API endpoints.
 */

document.addEventListener('DOMContentLoaded', () => {
    // State
    let currentLanguage = 'en';
    let currentHorizon = 7;
    let historyData = [];
    let forecastData = null;
    let metricsData = null;

    // Bilingual Dictionary
    const i18n = {
        en: {
            title: "Ujjain Onion Price Prediction System",
            subtitle: "Official AGMARKNET Data | Ujjain APMC, Madhya Pradesh",
            langLabel: "🌐 Language:",
            horizonLabel: "Select Forecast Horizon:",
            btnFetch: "Forecast Prices",
            btnFetchLoading: "Calculating Forecast...",
            m1Title: "Latest Mandi Price",
            m1Sub: "100 kg (Quintal)",
            m2Title: "Forecasted Price",
            m3Title: "Est. 50kg Bag Price",
            m3Sub: "Standard Farmer Bag",
            m4Title: "Market Trend Direction",
            m4Sub: "Short-term Momentum",
            chartTitle: "Historical Prices & Future Forecast Horizon",
            chartInfo: "Loading verified 2024–2026 Ujjain market trend...",
            tableTitle: "Forecasted Day-by-Day Schedule",
            colStep: "Step",
            colDate: "Forecast Date",
            colPriceQ: "Quintal Price (₹ / Q)",
            colPriceBag: "50kg Bag Price (₹ / 50kg)",
            colRange: "Estimated Uncertainty Range",
            benchmarkTitle: "📊 Model Evaluation Benchmarks (Unseen 2026 Test Period)",
            statusConnected: "API Connected",
            statusOffline: "API Offline",
            trendBullish: "🔴 Bullish (Spike)",
            trendBearish: "🟢 Bearish (Drop)",
            trendStable: "🟡 Stable"
        },
        hi: {
            title: "उज्जैन प्याज भाव पूर्वानुमान प्रणाली",
            subtitle: "आधिकारिक एगमार्कनेट डेटा | उज्जैन मंडी, मध्य प्रदेश",
            langLabel: "🌐 भाषा:",
            horizonLabel: "पूर्वानुमान समयावधि चुनें:",
            btnFetch: "भाव पूर्वानुमान लगाएं",
            btnFetchLoading: "गणना की जा रही है...",
            m1Title: "नवीनतम मंडी भाव",
            m1Sub: "100 किलोग्राम (क्विंटल)",
            m2Title: "अनुमानित क्विंटल भाव",
            m3Title: "अनुमानित 50kg बोरी भाव",
            m3Sub: "मानक किसान बोरी (50 किग्रा)",
            m4Title: "बाजार रुख (ट्रेंड)",
            m4Sub: "अल्पकालिक दिशा",
            chartTitle: "ऐतिहासिक मंडी भाव एवं आगामी अनुमानित ट्रेंड",
            chartInfo: "सत्यापित 2024-2026 उज्जैन मंडी डेटा प्रदर्शित...",
            tableTitle: "दैनिक पूर्वानुमान तालिका",
            colStep: "चरण (दिन)",
            colDate: "अनुमानित तारीख",
            colPriceQ: "क्विंटल भाव (₹ / Q)",
            colPriceBag: "50kg बोरी भाव (₹ / 50kg)",
            colRange: "अनुमानित दायरा (Uncertainty)",
            benchmarkTitle: "📊 मॉडल मूल्यांकन प्रदर्शन (अदृश्य 2026 परीक्षण काल)",
            statusConnected: "एपीआई कनेक्टेड",
            statusOffline: "ऑफलाइन",
            trendBullish: "🔴 मंदी का अनुमान (FALLING)",
            trendBearish: "🟢 तेजी का अनुमान (RISING)",
            trendStable: "🟡 स्थिर (STABLE)"
        }
    };

    // DOM Elements
    const elLangSelect = document.getElementById('lang-select');
    const elBtnGroup = document.getElementById('horizon-btn-group');
    const elBtnFetch = document.getElementById('btn-fetch');
    const elStatusBadge = document.getElementById('status-badge');
    const elStatusText = document.getElementById('status-text');

    // Attach Event Listeners
    elLangSelect.addEventListener('change', (e) => {
        currentLanguage = e.target.value;
        updateUIStrings();
        if (forecastData) renderDashboard(forecastData);
    });

    elBtnGroup.addEventListener('click', (e) => {
        if (e.target.classList.contains('btn-horizon')) {
            document.querySelectorAll('.btn-horizon').forEach(btn => btn.classList.remove('active'));
            e.target.classList.add('active');
            currentHorizon = parseInt(e.target.getAttribute('data-horizon'));
            fetchForecast(currentHorizon);
        }
    });

    elBtnFetch.addEventListener('click', () => {
        fetchForecast(currentHorizon);
    });

    // Initialize App
    init();

    async function init() {
        updateUIStrings();
        await checkHealth();
        await fetchHistory();
        await fetchMetrics();
        await fetchForecast(currentHorizon);
    }

    // Language Text Updater
    function updateUIStrings() {
        const t = i18n[currentLanguage];
        document.getElementById('title-text').innerText = t.title;
        document.getElementById('subtitle-text').innerText = t.subtitle;
        document.getElementById('lang-label').innerText = t.langLabel;
        document.getElementById('horizon-label').innerText = t.horizonLabel;
        document.getElementById('btn-fetch-text').innerText = t.btnFetch;
        document.getElementById('m1-title').innerText = t.m1Title;
        document.getElementById('m1-sub').innerText = t.m1Sub;
        document.getElementById('m2-title').innerText = t.m2Title;
        document.getElementById('m3-title').innerText = t.m3Title;
        document.getElementById('m3-sub').innerText = t.m3Sub;
        document.getElementById('m4-title').innerText = t.m4Title;
        document.getElementById('m4-sub').innerText = t.m4Sub;
        document.getElementById('chart-title').innerText = t.chartTitle;
        document.getElementById('chart-info').innerText = t.chartInfo;
        document.getElementById('table-title').innerText = t.tableTitle;
        document.getElementById('benchmark-title').innerText = t.benchmarkTitle;
    }

    // Health Check API
    async function checkHealth() {
        try {
            const res = await fetch('/api/health');
            const data = await res.json();
            if (data.status === 'healthy') {
                elStatusBadge.className = 'status-badge healthy';
                elStatusText.innerText = `${i18n[currentLanguage].statusConnected} (${data.latest_dataset_date})`;
            }
        } catch (err) {
            elStatusBadge.className = 'status-badge';
            elStatusText.innerText = i18n[currentLanguage].statusOffline;
        }
    }

    // Fetch History Data
    async function fetchHistory() {
        try {
            const res = await fetch('/api/history?limit=60');
            const data = await res.json();
            historyData = data.records || [];
        } catch (err) {
            console.error("Failed to fetch history:", err);
        }
    }

    // Fetch Metrics Data
    async function fetchMetrics() {
        try {
            const res = await fetch('/api/metrics');
            metricsData = await res.json();
            renderMetricsCard(metricsData);
        } catch (err) {
            console.error("Failed to fetch metrics:", err);
        }
    }

    // Main Forecast API Call
    async function fetchForecast(horizon) {
        setLoading(true);
        try {
            const res = await fetch(`/api/predict?horizon=${horizon}`);
            forecastData = await res.json();
            renderDashboard(forecastData);
        } catch (err) {
            alert("Failed to fetch forecast. Make sure FastAPI server is running.");
        } finally {
            setLoading(false);
        }
    }

    function setLoading(isLoading) {
        if (isLoading) {
            elBtnFetch.disabled = true;
            document.getElementById('btn-fetch-text').innerText = i18n[currentLanguage].btnFetchLoading;
        } else {
            elBtnFetch.disabled = false;
            document.getElementById('btn-fetch-text').innerText = i18n[currentLanguage].btnFetch;
        }
    }

    // Render All Cards, Charts & Table
    function renderDashboard(data) {
        const t = i18n[currentLanguage];
        const latestPrice = data.latest_actual_price;
        const results = data.daily_forecasts || [];
        const targetPred = results[results.length - 1];
        const targetPrice = targetPred ? targetPred.predicted_price : data.predicted_price;
        const bagPrice = targetPrice / 2.0;

        // Diff and trend calculation
        const diff = targetPrice - latestPrice;
        const pct = (diff / latestPrice) * 100;

        let trendText = t.trendStable;
        let badgeClass = 'metric-badge stable';
        if (pct > 0.5) {
            trendText = t.trendBearish; // Rising price
            badgeClass = 'metric-badge bearish';
        } else if (pct < -0.5) {
            trendText = t.trendBullish; // Falling price
            badgeClass = 'metric-badge bullish';
        }

        // 1. KPI Cards
        document.getElementById('m1-value').innerText = `₹ ${latestPrice.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;
        document.getElementById('m2-value').innerText = `₹ ${targetPrice.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;
        
        const m2Badge = document.getElementById('m2-badge');
        m2Badge.innerText = `${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`;
        m2Badge.className = badgeClass;

        document.getElementById('m3-value').innerText = `₹ ${bagPrice.toLocaleString('en-IN', {minimumFractionDigits: 2})}`;
        document.getElementById('m4-value').innerText = trendText;

        // 2. Render Plotly Chart
        renderPlotlyChart(historyData, results);

        // 3. Render Table
        renderTable(results);
    }

    // Render Plotly Interactive Chart
    function renderPlotlyChart(history, forecast) {
        const histDates = history.map(d => d.date);
        const histPrices = history.map(d => d.modal_price);

        const lastHistDate = histDates[histDates.length - 1];
        const lastHistPrice = histPrices[histPrices.length - 1];

        const predDates = [lastHistDate, ...forecast.map(f => f.forecast_date)];
        const predPrices = [lastHistPrice, ...forecast.map(f => f.predicted_price)];
        const upperBounds = [lastHistPrice, ...forecast.map(f => f.estimated_upper)];
        const lowerBounds = [lastHistPrice, ...forecast.map(f => f.estimated_lower)];

        const traceHist = {
            x: histDates,
            y: histPrices,
            mode: 'lines+markers',
            name: currentLanguage === 'hi' ? 'वास्तविक मंडी भाव' : 'Actual Mandi Price',
            line: { color: '#1E40AF', width: 2.5 },
            marker: { size: 4 }
        };

        const traceUpper = {
            x: predDates,
            y: upperBounds,
            mode: 'lines',
            line: { width: 0 },
            showlegend: false,
            hoverinfo: 'skip'
        };

        const traceLower = {
            x: predDates,
            y: lowerBounds,
            mode: 'lines',
            fill: 'tonexty',
            fillcolor: 'rgba(30, 64, 175, 0.15)',
            line: { width: 0 },
            name: currentLanguage === 'hi' ? 'अनुमानित अनिश्चितता सीमा' : 'Uncertainty Range'
        };

        const tracePred = {
            x: predDates,
            y: predPrices,
            mode: 'lines+markers',
            name: currentLanguage === 'hi' ? 'पूर्वानुमानित भाव' : 'Forecast Horizon',
            line: { color: '#DC2626', width: 3, dash: 'dash' },
            marker: { size: 6, color: '#DC2626' }
        };

        const layout = {
            margin: { l: 50, r: 20, t: 20, b: 40 },
            hovermode: 'x unified',
            legend: { orientation: 'h', y: 1.1, x: 0 },
            xaxis: { title: currentLanguage === 'hi' ? 'तारीख (Date)' : 'Date' },
            yaxis: { title: currentLanguage === 'hi' ? 'भाव (₹ / क्विंटल)' : 'Price (₹ / Quintal)' }
        };

        Plotly.newPlot('plotly-chart', [traceHist, traceUpper, traceLower, tracePred], layout, {responsive: true});
    }

    // Render Data Table
    function renderTable(results) {
        const tbody = document.getElementById('table-body');
        tbody.innerHTML = '';

        results.forEach(row => {
            const bagPrice = (row.predicted_price / 2.0).toFixed(2);
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>Day ${row.step}</strong></td>
                <td>${row.forecast_date}</td>
                <td><strong>₹ ${row.predicted_price.toLocaleString('en-IN', {minimumFractionDigits: 2})}</strong> / Q</td>
                <td>₹ ${parseFloat(bagPrice).toLocaleString('en-IN', {minimumFractionDigits: 2})} / 50kg</td>
                <td>₹ ${row.estimated_lower.toLocaleString('en-IN')} – ₹ ${row.estimated_upper.toLocaleString('en-IN')}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    // Render Metrics Footer Card
    function renderMetricsCard(data) {
        const el = document.getElementById('metrics-summary');
        if (!data || !data.test_metrics) return;

        const m = data.test_metrics;
        el.innerHTML = `
            <div style="display: flex; gap: 2rem; flex-wrap: wrap; margin-top: 0.5rem;">
                <div><strong>Selected Model:</strong> ${data.selected_model}</div>
                <div><strong>2026 Test MAE:</strong> ₹ ${m.MAE.toFixed(2)} / Q</div>
                <div><strong>RMSE:</strong> ₹ ${m.RMSE.toFixed(2)} / Q</div>
                <div><strong>R² Score:</strong> ${m.R2.toFixed(4)}</div>
                <div><strong>MAPE:</strong> ${m['MAPE_%'].toFixed(2)}%</div>
            </div>
        `;
    }
});
