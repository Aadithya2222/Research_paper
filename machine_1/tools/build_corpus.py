"""Build local synthetic corpus for deterministic retrieval and offline reproducibility.

Generates tasks/corpus.json containing domain documents for all 50 benchmark topics,
ensuring tasks can be solved deterministically via retrieval without open web access.
"""
from __future__ import annotations

import json
from pathlib import Path

# Complete document database covering all 50 benchmark topics with exact factual phrases
DOMAIN_DOCUMENTS = [
    # 1. Agriculture / Precision Irrigation
    {"doc_id": "D001", "title": "Precision Irrigation", "text": "Precision irrigation uses real-time soil-moisture sensor data placed at 15cm and 45cm root depth. It achieves 25 to 40 percent water reduction and application efficiency to 95 percent with drip emitters."},
    # 2. Automotive / EV Batteries
    {"doc_id": "D002", "title": "Electric-Vehicle Batteries", "text": "Electric-vehicle batteries capture kinetic energy during regenerative braking, converting it to direct current. LFP offers 270°C thermal stability, while NMC provides 250 Wh/kg energy density."},
    # 3. Aviation / Aircraft Turnaround
    {"doc_id": "D003", "title": "Aircraft Turnaround", "text": "Aircraft turnaround time is measured between chocks on and chocks off. Turnaround causes 35 percent of airline delays. Parallel deboarding saves 8 to 12 minutes, cutting apron congestion by 18 percent."},
    # 4. Banking / Credit Risk
    {"doc_id": "D004", "title": "Credit Risk", "text": "Credit risk scoring uses numerical credit scores from 300 to 850. Payment history carries a 35% weight and credit utilization carries 30%. ML models require SHAP analysis for adverse action explanations."},
    # 5. Insurance / Claims Fraud
    {"doc_id": "D005", "title": "Claims Fraud", "text": "Insurance claims fraud involves intentional deception accounting for 10 percent of operating losses. Rule-based detection yields 40% false positives, whereas ML reduces false positives by 25 percent."},
    # 6. Healthcare Operations / Hospital Bed Management
    {"doc_id": "D006", "title": "Hospital Bed Management", "text": "Hospital bed management tracks occupied inpatient beds. Occupancy over 85 percent causes boarding. Predictive bed management maintains target occupancy below 82 percent and cuts wait times by 45 minutes."},
    # 7. Pharmaceuticals / Clinical Trials
    {"doc_id": "D007", "title": "Clinical Trials", "text": "Clinical trials use randomized controlled trials. Phase II tests 100-300 patients for efficacy, while Phase III evaluates 1,000-3,000 patients across multi-center trial sites."},
    # 8. Biotechnology / CRISPR Ethics
    {"doc_id": "D008", "title": "CRISPR Ethics", "text": "CRISPR-Cas9 enables site-specific gene editing. Somatic gene editing is non-heritable, whereas germline editing creates inherited genetic alterations subject to regulatory scrutiny."},
    # 9. Cybersecurity / Phishing Defense
    {"doc_id": "D009", "title": "Phishing Defense", "text": "Phishing defense against domain spoofing relies on SPF/DKIM/DMARC authentication, secure email gateways, and simulated user training."},
    # 10. Cloud Computing / Autoscaling
    {"doc_id": "D010", "title": "Autoscaling", "text": "Cloud autoscaling adjusts active instances based on CPU utilization and request latency using horizontal scaling and vertical scaling."},
    # 11. Semiconductors / Chip Fabrication Yield
    {"doc_id": "D011", "title": "Chip Fabrication Yield", "text": "Semiconductor manufacturing yield represents non-defective dies produced on a silicon wafer, calculated from defect density and total die area."},
    # 12. Telecommunications / 5G Network Slicing
    {"doc_id": "D012", "title": "5G Network Slicing", "text": "5G network slicing creates logical sub-networks for eMBB broadband, URLLC ultra-low latency under 1ms, and mMTC massive IoT connections."},
    # 13. Renewable Energy / Solar PV
    {"doc_id": "D013", "title": "Solar PV", "text": "Photovoltaic module efficiency averages 20 to 23 percent. Utility-scale solar achieves $35/MWh levelized cost of energy compared to residential rooftop systems."},
    # 14. Wind Energy / Offshore Wind
    {"doc_id": "D014", "title": "Offshore Wind", "text": "Onshore wind turbines achieve 30 to 40 percent capacity factor. Offshore wind turbines reach 45 to 55 percent capacity factor but incur higher capital expenditure."},
    # 15. Oil and Gas / Methane Emissions
    {"doc_id": "D015", "title": "Methane Emissions", "text": "Methane possesses over 80 times global warming potential compared to CO2. Continuous optical gas imaging sensors detect leaks faster than manual sniffer surveys."},
    # 16. Nuclear Energy & Security / Fission and Safeguards
    {"doc_id": "D016", "title": "Fission and Safeguards", "text": "Civilian light-water reactors use low-enriched Uranium-235. IAEA international safeguards apply material accounting to prevent diversion to weapons-grade non-proliferation enrichment."},
    # 17. Construction / Building Information Modeling
    {"doc_id": "D017", "title": "Building Information Modeling", "text": "Building Information Modeling (BIM) uses 3D parametric models. 3D clash detection cuts field rework costs by up to 20 percent."},
    # 18. Architecture / Passive Building Design
    {"doc_id": "D018", "title": "Passive Building Design", "text": "Passive building design optimizes orientation, daylighting, thermal insulation, and passive solar gain to reduce heating and cooling loads."},
    # 19. Real Estate / Property Valuation
    {"doc_id": "D019", "title": "Property Valuation", "text": "Property valuation combines comparable-sales approach, income capitalization approach, and cost approach to estimate fair market value."},
    # 20. Logistics / Last-Mile Delivery
    {"doc_id": "D020", "title": "Last-Mile Delivery", "text": "Last-mile delivery accounts for 53 percent of total shipping cost. Urban micro-fulfillment centers reduce delivery radius and transit times."},
    # 21. Supply Chain / Inventory Resilience
    {"doc_id": "D021", "title": "Inventory Resilience", "text": "Inventory resilience balances just-in-time logistics with safety stock buffers to mitigate supply chain disruptions and lead-time variability."},
    # 22. Retail / Dynamic Pricing
    {"doc_id": "D022", "title": "Dynamic Pricing", "text": "Dynamic pricing algorithms adjust price levels based on demand elasticity, competitor pricing, and inventory clearance velocity."},
    # 23. E-commerce / Recommendation Systems
    {"doc_id": "D023", "title": "Recommendation Systems", "text": "Recommendation systems combine collaborative filtering and content-based filtering to optimize user conversion and catalog discovery."},
    # 24. Manufacturing / Predictive Maintenance
    {"doc_id": "D024", "title": "Predictive Maintenance", "text": "Predictive maintenance uses vibration and thermal IoT sensors to predict equipment failure before catastrophic breakdown."},
    # 25. Robotics / Warehouse Robots
    {"doc_id": "D025", "title": "Warehouse Robots", "text": "Autonomous mobile robots (AMRs) optimize warehouse material handling, fleet dispatching, and dynamic obstacle avoidance."},
    # 26. Food Processing / HACCP
    {"doc_id": "D026", "title": "HACCP", "text": "Hazard Analysis Critical Control Point (HACCP) identifies biological, chemical, and physical food safety hazards at critical control points."},
    # 27. Hospitality / Hotel Revenue Management
    {"doc_id": "D027", "title": "Hotel Revenue Management", "text": "Hotel revenue management optimizes room rate yield using occupancy rate, Average Daily Rate (ADR), and Revenue Per Available Room (RevPAR)."},
    # 28. Tourism / Overtourism Management
    {"doc_id": "D028", "title": "Overtourism Management", "text": "Overtourism management balances visitor capacity with resident quality of life using timed entry permits, tourist taxes, and visitor caps."},
    # 29. Education / Adaptive Learning
    {"doc_id": "D029", "title": "Adaptive Learning", "text": "Adaptive learning software dynamically adjusts instruction pacing and difficulty based on student item response theory performance."},
    # 30. Higher Education / Student Retention
    {"doc_id": "D030", "title": "Student Retention", "text": "Student retention analytics identify early academic disengagement indicators to target financial aid and tutoring interventions."},
    # 31. Legal Services / Contract Review AI
    {"doc_id": "D031", "title": "Contract Review AI", "text": "Contract review AI accelerates non-disclosure agreement and commercial contract risk extraction, reducing legal review turnaround time."},
    # 32. Public Administration / Digital Identity
    {"doc_id": "D032", "title": "Digital Identity", "text": "Digital identity architectures balance biometric authentication, zero-knowledge proofs, and privacy compliance across public services."},
    # 33. Accounting / Revenue Recognition
    {"doc_id": "D033", "title": "Revenue Recognition", "text": "Revenue recognition under ASC 606 requires identifying performance obligations, transaction price, and transfer of control."},
    # 34. Auditing / Audit Sampling
    {"doc_id": "D034", "title": "Audit Sampling", "text": "Audit sampling evaluates population financial statement accuracy using statistical sampling, tolerable misstatement thresholds, and stratification."},
    # 35. Fintech / Digital Payments Fraud
    {"doc_id": "D035", "title": "Digital Payments Fraud", "text": "Digital payments fraud detection combines device fingerprinting, behavioral biometrics, and real-time velocity checking rules."},
    # 36. Blockchain / Smart Contracts
    {"doc_id": "D036", "title": "Smart Contracts", "text": "Smart contracts execute immutable business logic on distributed ledgers, offering cryptographic verification without intermediaries."},
    # 37. Economics / Inflation
    {"doc_id": "D037", "title": "Inflation", "text": "Inflation measures consumer price index growth, distinguishing demand-pull inflation from cost-push supply chain shock inflation."},
    # 38. Macroeconomic Policy / Interest Rates
    {"doc_id": "D038", "title": "Interest Rates", "text": "Central bank policy interest rates influence commercial borrowing costs, money supply, employment targets, and exchange rates."},
    # 39. Development Economics / Microfinance
    {"doc_id": "D039", "title": "Microfinance", "text": "Microfinance provides microloans and financial services to low-income entrepreneurs using group lending solidarity collateral models."},
    # 40. Environmental Science / Air Pollution
    {"doc_id": "D040", "title": "Air Pollution", "text": "Air pollution monitoring measures fine particulate matter PM2.5 concentrations from combustion sources and industrial emissions."},
    # 41. Climate Science / Carbon Budgets
    {"doc_id": "D041", "title": "Carbon Budgets", "text": "Carbon budgets quantify total cumulative CO2 emissions allowable to restrict global warming within 1.5°C or 2.0°C climate targets."},
    # 42. Water Utilities / Non-Revenue Water
    {"doc_id": "D042", "title": "Non-Revenue Water", "text": "Non-revenue water measures water lost before reaching customer meters due to distribution pipe leaks and commercial theft."},
    # 43. Waste Management / Recycling Systems
    {"doc_id": "D043", "title": "Recycling Systems", "text": "Recycling systems compare single-stream automated sorting facilities with source-separated municipal recycling programs."},
    # 44. Mining / Mine Safety
    {"doc_id": "D044", "title": "Mine Safety", "text": "Mine safety engineering enforces underground ventilation controls, methane monitoring, and ground support rock bolting."},
    # 45. Metallurgy / Heat Treatment
    {"doc_id": "D045", "title": "Heat Treatment", "text": "Heat treatment modifies metal mechanical properties using controlled heating, quenching, and tempering microstructural processes."},
    # 46. Chemicals / Process Safety
    {"doc_id": "D046", "title": "Process Safety", "text": "Chemical process safety applies safety data sheets (SDS), safety instrumented systems, and quantitative hazard evaluation."},
    # 47. Textiles / Sustainable Dyeing
    {"doc_id": "D047", "title": "Sustainable Dyeing", "text": "Sustainable dyeing technologies use supercritical carbon dioxide waterless dyeing processes to eliminate toxic wastewater effluent."},
    # 48. Fashion / Circular Economy
    {"doc_id": "D048", "title": "Circular Economy", "text": "Circular fashion promotes garment repair, garment rental, textile-to-textile recycling, and closed-loop material design."},
    # 49. Media / Recommendation Algorithms
    {"doc_id": "D049", "title": "Recommendation Algorithms", "text": "Media recommendation algorithms balance user engagement optimization with algorithmic feed diversity and misinformation guardrails."},
    # 50. Journalism / Source Verification
    {"doc_id": "D050", "title": "Source Verification", "text": "Source verification in investigative journalism evaluates primary documentary evidence, eyewitness credibility, and digital forensic provenance."}
]


def build_full_corpus() -> List[dict]:
    return list(DOMAIN_DOCUMENTS)


def main() -> int:
    out_path = Path("tasks/corpus.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    corpus = {"documents": build_full_corpus()}
    out_path.write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(corpus['documents'])} documents to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
