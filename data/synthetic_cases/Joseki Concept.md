## **New Concept Layer: Open Wrapper Infrastructure**

Joseki is premised on the following observation: many AI products encode much of their value in the way they instruct, constrain, and contextualize foundation models.

This statement is not meant to belittle wrapper-esque products. A good instruction layer can represent years of domain knowledge and expertise. However, that domain expertise may also be highly condensable and portable.

## **Joseki’s Market Thesis**

The deeper thesis is that many profit-generating institutions, research programs, professional services, educational products, media workflows, legal and compliance processes, consulting playbooks, software tools, and internal business operations will become, at least in part, structured instruction systems for LLMs. 

We see this as a massive opportunity. As foundation models become more capable and more interchangeable, the durable value may increasingly come from the layers around the model: instructions, examples, domain-specific data packs, safety constraints, and trust.

The future may not be thousands of isolated AI applications with non-portable workflows locked inside each product. It may be a public and private ecosystem of reusable AI work products where creators compete on quality, trust, relevance, evidence, portability, and maintenance.

Joseki will be the forum to build, and exchange these layers.

This is the optimistic version of the wrapper thesis: AI lets us hyperscale expertise not only by transmitting text, but by transmitting structured ideas. A good Joseki package is not merely a block of words. It is a compressed unit of know-how that can instruct different models toward useful outcomes. Joseki is infrastructure for moving on the idea plane, not only the text plane.

## **How does this improve on existing solutions?**

Many existing AI workflow tools sit between two incomplete models. Traditional prompt marketplaces often sell raw text without enough evidence that the prompt works, without clear versioning, without safety rules, without licensing clarity, and without a reliable way to distinguish a durable workflow from a clever one-off prompt. At the other end, many AI applications package useful instruction patterns inside proprietary interfaces, which can make the workflow harder to inspect, test, reuse, or move between model providers.

Joseki improves on both by treating the instruction layer as a real software artifact. A Joseki package would not only include a prompt. It would include a PromptSpec, SafetySpec, EvalSpec, LicenseSpec, Data Pack when necessary, and EvidencePack showing who verified that it works and under what model conditions.

The key differentiator is **model-agnostic portability**. Joseki is built for a world where users may move between OpenAI, Anthropic, Google, Meta, local open-source models, and future providers that do not exist yet. The workflow should not be trapped inside one interface or one model vendor. Joseki makes the valuable layer portable: the instructions, examples, tests, safety constraints, data dependencies, and verification history can travel across providers.

A prompt that worked well on one model may need adjustment on another. A workflow that performed reliably six months ago may degrade after a model update. Instead of relying on a weak signal like “Last Updated,” Joseki uses a **Liveness Score** based on real user reports: works on GPT-4o, needs adjustment on Claude, works on Gemini, works after the May 2026 model update, verified by 500 users this morning, and so on.

Joseki also recognizes that many valuable AI workflows are not only instructions. They are instructions plus current context. A legal analysis workflow, for example, may require a legal reasoning PromptSpec, a current Data Pack of 2026 tax code changes, benchmark questions, jurisdictional disclaimers, and evidence that practicing lawyers have tested it. This makes Joseki more than a prompt hub. It becomes a trusted registry for reusable AI work products where prompts, data, evaluations, safety rules, licenses, and verification move together.

In short, Joseki is needed because the AI economy is moving toward a world where foundation models become more capable, more interchangeable, and more widely available. As that happens, more value moves up the stack into the instruction, evaluation, data, and trust layers. Joseki’s opportunity is to make that layer visible, portable, governable, and useful.

## **Minimal Wrapper Extraction**

Given an output, codebase, workflow, or AI product, Joseki will infer the minimal instruction pattern needed for another LLM to reproduce something similar.

In this context, a **seed** is the smallest useful idea that can reliably steer an LLM toward a desired result. It is not necessarily the full prompt, the full product, or the full workflow. It is the compressed conceptual core: the role, goal, constraints, examples, evaluation criteria, and domain assumptions that make the output possible.

In plain English:

“Find the smallest portable idea that lets another model recreate this kind of result.”

This could help users compress complex workflows into reusable PromptSpecs. It could also help creators understand what makes a workflow valuable: not only the surface wording, but the underlying structure that guides the model.

## **Bottom Line**

These specs turn Joseki from a prompt marketplace into an AI workflow infrastructure layer.

Joseki is built on the insight that many AI products contain valuable instruction systems: prompts, examples, data, evals, safety policies, workflow logic, and trust artifacts built around foundation models.

A normal prompt marketplace sells text.

A closed AI workflow often makes that instruction layer difficult to inspect or reuse.

Joseki does the following: it makes the wrapper layer explicit, structured, portable, testable, licensable, auditable, and community-verified.

Every serious AI workflow should come with instructions, context, tests, safety rules, usage terms, installation steps, packaged data when needed, and evidence that it works. Joseki provides the structure for that ecosystem.

The long-term opportunity is bigger than prompt sharing. Joseki is infrastructure for model-agnostic AI work products: portable packages of expertise that can move between model providers, survive model updates, and let communities verify what actually works.

