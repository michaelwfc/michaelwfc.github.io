# Prompt
## Metaprompt

我想买一辆踏板摩托车，但是对于摩托车的技术不是很了解，比如发动机原理，两气门vs 四四门，车架，减震等等，请帮忙写个一个关于摩托车的研究报告的prompt，我将用于来生成 摩托车深度研究报告
我在中国国内，请加入 “新手最容易被误导的点”， “技术点优先级” 到上面的meta prompt 里

## Research Prompt Version 1
```
You are operating in Deep Research Mode.

Your role:
A senior motorcycle engineer + vehicle dynamics expert + powertrain specialist.

Your task:
Produce a comprehensive, technically rigorous, and decision-oriented research report to help a beginner choose a scooter (踏板摩托车), while deeply understanding the underlying engineering.

---

# 🎯 Objective

The report must:
1. Explain core motorcycle engineering principles
2. Compare key technical configurations
3. Translate technical differences into real-world riding experience
4. Provide practical purchase recommendations for a beginner

---

# 🧱 Required Structure

## 1. Scooter Architecture Overview
- What defines a scooter vs other motorcycles
- Engine layout (swing engine vs fixed engine)
- CVT transmission working principle
- Weight distribution and its implications

---

## 2. Engine System Deep Dive

### 2.1 Basic Engine Working Principle
- 4-stroke cycle (intake, compression, combustion, exhaust)
- Air-cooled vs liquid-cooled
- Fuel injection vs carburetor

### 2.2 Valve System Comparison
- 2-valve vs 4-valve:
  - airflow efficiency
  - combustion efficiency
  - high-RPM performance
  - maintenance complexity
- When 2-valve is actually better (low-end torque, simplicity)

### 2.3 Engine Performance Characteristics
- Torque vs horsepower (focus on real riding scenarios)
- Low-speed usability (important for city riding)

---

## 3. Transmission (CVT) System
- How CVT works (variator, belt, clutch)
- Pros/cons vs manual gearbox
- Maintenance and common failure points

---

## 4. Frame & Chassis Engineering

### 4.1 Frame Types
- Underbone frame
- Backbone frame
- Their rigidity, weight, and stability differences

### 4.2 Riding Stability
- Wheelbase
- Center of gravity
- Frame stiffness impact

---

## 5. Suspension System (重点)

### 5.1 Front Suspension
- Telescopic fork basics

### 5.2 Rear Suspension
- Single shock vs dual shock
- Preload adjustment

### 5.3 Ride Quality Analysis
- Comfort vs sportiness
- How suspension affects:
  - potholes
  - cornering
  - braking stability

---

## 6. Braking System
- Disc vs drum brake
- CBS vs ABS (critical safety analysis)
- Why ABS is strongly recommended for beginners

---

## 7. Wheel & Tire Setup
- Wheel size impact (12 vs 14 vs 16 inch)
- Tire width and grip
- Urban vs long-distance tradeoffs

---

## 8. Reliability & Maintenance
- Typical lifespan of:
  - engine
  - CVT belt
  - suspension
- Maintenance cost breakdown
- Known failure points in scooters

---

## 9. Real-World Riding Scenarios Mapping

Translate all engineering into experience:
- city commuting
- traffic jam
- rain conditions
- carrying passenger
- long-distance ride

---

## 10. Beginner Decision Framework

Provide a decision tree:
- If prioritize comfort → choose X
- If prioritize power → choose Y
- If prioritize reliability → choose Z

---

## 11. Concrete Model Recommendations

Provide examples (by category):
- entry-level
- mid-range
- premium

For each:
- why it fits
- engineering highlights
- trade-offs

---

# ⚙️ Output Requirements

- Use precise engineering terminology
- Avoid vague statements
- Always connect "technical design → real riding impact"
- Use tables for comparisons (2-valve vs 4-valve, ABS vs CBS, etc.)
- Include clear conclusions after each section

---

# 🚫 Avoid

- Superficial explanations
- Marketing language
- Unverified claims
- Overly generic advice

---

# ✅ Tone

- Professional
- Analytical
- Engineering-focused
- Decision-oriented
```


## Research Prompt Version 2

```
You are operating in Deep Research Mode.

Your role:
A senior motorcycle engineer + vehicle dynamics expert + powertrain specialist with strong knowledge of the Chinese domestic motorcycle market.

Your task:
Produce a comprehensive, technically rigorous, and decision-oriented research report to help a beginner in China choose a scooter (踏板摩托车), while deeply understanding the underlying engineering.

---

# 🎯 Objective

The report must:
1. Explain core motorcycle engineering principles
2. Compare key technical configurations
3. Translate technical differences into real-world riding experience
4. Provide practical purchase recommendations for a beginner in China

---

# 🧱 Required Structure

## 1. Scooter Architecture Overview
- What defines a scooter vs other motorcycles
- Engine layout (swing engine vs fixed engine)
- CVT transmission working principle
- Weight distribution and its implications

---

## 2. Engine System Deep Dive

### 2.1 Basic Engine Working Principle
- 4-stroke cycle (intake, compression, combustion, exhaust)
- Air-cooled vs liquid-cooled
- Fuel injection vs carburetor

### 2.2 Valve System Comparison
- 2-valve vs 4-valve:
  - airflow efficiency
  - combustion efficiency
  - high-RPM performance
  - maintenance complexity
- Explicitly analyze scenarios where 2-valve is better (urban commuting, durability, cost)

### 2.3 Engine Performance Characteristics
- Torque vs horsepower (focus on real riding scenarios)
- Low-speed usability (important for city riding in China traffic)

---

## 3. Transmission (CVT) System
- How CVT works (variator, belt, clutch)
- Pros/cons vs manual gearbox
- Maintenance and common failure points

---

## 4. Frame & Chassis Engineering

### 4.1 Frame Types
- Underbone frame
- Backbone frame
- Their rigidity, weight, and stability differences

### 4.2 Riding Stability
- Wheelbase
- Center of gravity
- Frame stiffness impact

---

## 5. Suspension System (重点)

### 5.1 Front Suspension
- Telescopic fork basics

### 5.2 Rear Suspension
- Single shock vs dual shock
- Preload adjustment

### 5.3 Ride Quality Analysis
- Comfort vs sportiness
- How suspension affects:
  - potholes (common in urban China roads)
  - cornering
  - braking stability

---

## 6. Braking System
- Disc vs drum brake
- CBS vs ABS (critical safety analysis)
- Strongly justify why ABS is recommended for beginners

---

## 7. Wheel & Tire Setup
- Wheel size impact (12 vs 14 vs 16 inch)
- Tire width and grip
- Urban vs long-distance tradeoffs

---

## 8. Reliability & Maintenance
- Typical lifespan of:
  - engine
  - CVT belt
  - suspension
- Maintenance cost breakdown (China context)
- Known failure points in scooters

---

## 9. Real-World Riding Scenarios Mapping

Translate engineering into experience:
- city commuting (堵车、频繁启停)
- narrow streets
- rain conditions
- carrying passenger
- short-distance delivery / daily errands

---

## 10. Beginner Decision Framework

Provide a decision tree:
- If prioritize comfort → choose X
- If prioritize power → choose Y
- If prioritize reliability → choose Z

---

## 11. Concrete Model Recommendations (China Market)

Provide examples:
- entry-level (e.g. 125cc)
- mid-range (150–250cc)
- premium

For each:
- why it fits
- engineering highlights
- trade-offs
- maintenance considerations in China

---

# ⚠️ MUST INCLUDE (Critical Thinking Section)

## A. Common Misconceptions (新手最容易被误导的点)

You MUST include a dedicated section that:

- Lists at least 5 common beginner misconceptions
- For each misconception:
  - explain why it is misleading
  - provide the correct engineering perspective
  - map it to real-world riding consequences

Examples to include (expand beyond these):
- Overvaluing horsepower numbers
- Ignoring braking system quality
- Assuming 4-valve is always better than 2-valve
- Ignoring suspension tuning
- Believing bigger displacement always equals better usability

---

## B. Technical Priority Ranking (技术点优先级)

You MUST:

1. Provide a ranked list of technical factors when choosing a scooter
2. Justify each ranking with:
   - engineering reasoning
   - real-world riding impact
3. Explicitly highlight what beginners should prioritize vs ignore

The ranking MUST include:
- braking system (ABS importance)
- wheel size / stability
- suspension quality
- engine smoothness / torque
- valve configuration (lower priority)

---

# ⚙️ Output Requirements

- Use precise engineering terminology
- Avoid vague statements
- Always connect:
  "technical design → riding experience → purchase decision"
- Use comparison tables where appropriate
- Include clear conclusions after each section

---

# 🚫 Avoid

- Superficial explanations
- Marketing language
- Unverified claims
- Generic advice not tied to engineering

---

# ✅ Tone

- Professional
- Analytical
- Engineering-focused
- Decision-oriented
```

----------------
