# 🎙️ Video Presentation Script — LPDG Gateway Health Prediction

> **Presenter:** Keerthi Machanooru ([@Keerthimk24](https://github.com/Keerthimk24))  
> **Topic:** LPDG Gateway Health Prediction Intelligence Dashboard & Cost-Optimized ML Pipeline  
> **Duration:** 6 minutes 55 seconds  
> **Voiceover Script:** [ELEVENLABS_SCRIPT.txt](ELEVENLABS_SCRIPT.txt) (with exact `<break time="..." />` SSML pause tags)  
> **Video Recording:** [Watch on Google Drive](https://drive.google.com/file/d/1wDi-DuQ4l0REjnqG_6AtXIik7fPECC9h/view?usp=drive_link)

---

## ⏱️ Minute-by-Minute Outline

| Timestamp | Section | Visual On Screen | Key Discussion Points |
| :---: | :--- | :--- | :--- |
| **0:00 – 0:50** | **Executive Introduction** | Dashboard Header & Title | Utility background, 320 gateways, €600 non-reading penalty vs €380 visit cost, 15 visits/week budget |
| **0:50 – 2:05** | **Fleet Health Scorecard** | KPI Cards (Top Banner) | 302 Safe vs 30 Broken (91% / 9%), €30,380 net savings, 0.888 AUC-ROC, 267% catch improvement, 120 total visits |
| **2:05 – 3:35** | **Visual Analytics Section** | 4 Operational Charts | • Maintenance Accuracy (AI vs 3σ)<br>• Weekly Operating Cost (€38.7k down to €30.9k)<br>• Why Gateways Break (42% cellular, 28% packet loss, 18% hardware, 12% power)<br>• Chronic Problem Gateways (Top 10 repeat offenders) |
| **3:35 – 5:00** | **Dispatch Schedule** | Interactive Priority Table | 15 prioritized gateways per week, calibrated risk score (0–100), actionable field instructions, under-300-char explanations, search & week filter |
| **5:00 – 6:10** | **Next Week Early Warning** | Watchlist (Ranks 16–23) | Proactive maintenance: route batching (adjacent towers) and zero-cost remote modem reboots |
| **6:10 – 6:55** | **Summary & Conclusion** | Production Architecture | +67% more broken gateways caught, false alarms eliminated, €30k+ savings, single-command deployment |

---

## 📜 Full Spoken Script (with Timestamps)

### [0:00 – 0:50] 1. Executive Introduction & Operational Challenge
> **Visual:** Operations Dashboard top header: *"LPDG Gateway Health Prediction — Operations Dashboard"*.

"Hello everyone, my name is Keerthi Machanooru. Welcome to this walkthrough of the LPDG Gateway Health Prediction Intelligence Dashboard.

In our municipal utility network, approximately 320 LoRaWAN gateways collect and relay data from thousands of smart water and gas meters. When a gateway silently fails in the field, meter data stops flowing, costing the utility 600 euros per week in regulatory non-reading penalties.

Our field engineering team faces a strict operational limit: we can only dispatch technicians to 15 gateways per week, and each visit costs 380 euros.

In the past, operations relied on spreadsheets and basic three-sigma statistical anomaly thresholds. That resulted in a 61% false alarm rate, where technicians repeatedly drove out only to find completely healthy gateways.

This dashboard presents our machine learning solution: a cost-weighted system that directly optimizes financial returns and tells field teams exactly which 15 gateways to visit each week."

---

### [0:50 – 2:05] 2. Fleet Health Scorecard (KPIs)
> **Visual:** Top KPI metric cards: *Fleet Reliability Ratio*, *Net Cost Savings*, *Precision & Recall*, and *Evaluation Dispatches*.

"Looking at the top of our dashboard, the Fleet Health Scorecard gives management an instant executive summary of the entire network.

First, our fleet reliability ratio stands at 302 Safe Gateways, or 91%, versus 30 Broken Gateways, or 9%. The AI continuously filters out healthy noise so our field engineers focus exclusively on genuine risks.

Next, across the 4-week February evaluation period, our system delivers 30 thousand, 380 euros in cumulative net savings compared to the statistical baseline.

In terms of accuracy, our model achieves a 0.888 AUC-ROC. Each week, we successfully catch 11 out of 60 bad gateways, sending only 4 false alarms. That is compared to the baseline, which caught only 3 bad gateways and sent 12 false alarms. That represents a 267% improvement in broken gateways caught, while reducing wasted technician travel costs by over 3 thousand euros every single week.

Across the entire 8-week evaluation window, our system generates exactly 120 prioritized visit recommendations, perfectly adhering to the 15 visits per week constraint."

---

### [2:05 – 3:35] 3. Visual Analytics: 4 Operational Charts
> **Visual:** Scrolling down to the 4 visual analytics charts side-by-side.

"Scrolling down to our visual analytics section, four intuitive charts translate complex telemetry patterns into actionable operational intelligence.

**Chart 1: Maintenance Accuracy: Gateways Caught versus Wasted Trips.**  
Here we see our AI model side-by-side with the three sigma baseline. While the baseline wastes 80% of its dispatches on false alarms, our cost-weighted model directs nearly three-quarters of all engineer visits directly to failing hardware.

**Chart 2: Weekly Operating Cost.**  
This chart plots total weekly cost, combining 380 euro visit fees and 600 euro weekly penalties for missed gateways. Our model consistently lowers network operating costs from 38 thousand, 760 euros per week down to 30 thousand, 920 euros per week, generating steady weekly savings of nearly 7 thousand, 800 euros every week.

**Chart 3: Why Gateways Break, showing the failure root causes breakdown.**  
Rather than treating all failures as generic anomalies, our feature pipeline diagnoses the root cause. As shown in this donut chart:
- 42% of issues are caused by cellular backhaul disconnects.
- 28% stem from smart meter packet dropouts.
- 18% are due to repeat hardware defects.
- And 12% are chronic power reboot brownouts.

**Chart 4: Chronic Problem Gateways.**  
This horizontal bar chart highlights the top 10 repeat offender gateways flagged across 7 to 8 weeks. This tells operations that repeatedly rebooting these units is a waste of money; they need a permanent physical hardware swap."

---

### [3:35 – 5:00] 4. Priority Dispatch Schedule (The 15 Visits/Week Table)
> **Visual:** Interactive 15-gateway dispatch schedule table with Search and Week Filter dropdown.

"Next, we come to the core operational tool: the Complete 8-Week Visit Dispatch Schedule.

This table lists the exact 15 gateways selected for each Monday week, ranked from most critical to least critical.

Notice the interactive controls at the top. The Week Filter dropdown allows dispatch managers to navigate between all 8 weeks, from February 2nd through March 23rd. When we switch weeks, the table dynamically refreshes with the updated queue for that period.

We also have a live Search bar. If a dispatcher wants to find a specific gateway hex ID or filter by problem type, such as typing 'antenna' or 'reboot', the table filters instantaneously.

For every gateway, the table provides three vital pieces of decision support:
1. **Calibrated Risk Score out of 100:** Ranging from high-90s emergency failures down to elevated risks, giving dispatchers an immediate sense of urgency.
2. **Recommended Field Action:** This is critical for field teams. Instead of arriving blind, the technician knows whether to bring a replacement antenna, swap the gateway unit, or inspect power lines.
3. **Diagnosed Reason:** A concise explanation under 300 characters explaining the exact statistical drivers behind the recommendation."

---

### [5:00 – 6:10] 5. Next Week Early Warning Watchlist (Ranks 16–23)
> **Visual:** Amber-tinted Watchlist table at the bottom of the dashboard.

"Scrolling further down, we reach a unique proactive feature: the Next Week Early Warning Watchlist, highlighting gateways ranked number 16 to number 23.

Because our weekly budget is strictly capped at 15 visits, these gateways just missed the current week's dispatch cutoff, but our model has detected early degradation trends.

This gives operations two huge advantages:

**First: Geographic Route Batching.** If Gateway 18 is located on the same street or cellular tower as Gateway 3, operations can instruct the technician to service both during the same trip, eliminating a separate truck roll next week.

**Second: Zero-Cost Remote Interventions.** Before sending a physical technician, network engineers can attempt over-the-air firmware updates or remote modem power cycles to resolve issues before physical hardware failure occurs."

---

### [6:10 – 6:55] 6. Conclusion & Production Readiness
> **Visual:** Full dashboard overview, terminal quickstart commands, and GitHub repository.

"To wrap up:

This dashboard transforms raw IoT telemetry into an automated, cost-optimized predictive maintenance workflow.

By replacing guesswork with machine learning:
- We catch 67% more broken gateways.
- We eliminate wasted false alarm trips.
- And we deliver over 30 thousand euros in real financial savings to the utility.

The entire dashboard is fully interactive, runs locally in a single command, and is ready for production deployment.

Thank you very much for watching!"
