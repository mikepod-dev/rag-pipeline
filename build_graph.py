import json

import networkx as nx

from pipeline import all_chunks, ask_llm


def extract_triples(text):
    prompt = f"""Extract factual relationships from this text as a list of triples: (subject, relationship, object).

Only extract clear, factual relationships explicitly stated in the text - do not infer or guess. Use short, consistent entity names (e.g., "Dog" not "the domestic dog").

Text: {text}

Reply with ONLY a JSON list of triples, like: [["Dog", "descended_from", "Wolf"], ["Coffee", "contains", "Caffeine"]]
If no clear relationships exist, reply with an empty list: []"""

    response = ask_llm(prompt, [])
    try:
        cleaned = response.strip().strip("```json").strip("```").strip()
        triples = json.loads(cleaned)
        return triples
    except json.JSONDecodeError:
        return []


graph = nx.DiGraph()

target_sources = [
    "dog.txt",
    "wiki_dog.txt",
    "wiki_wolf.txt",
    "wiki_cat.txt",
    "animals_overview.txt",
    "wiki_domestication.txt",
]
chunks_per_source = 10
sample_chunks = []
for source in target_sources:
    source_chunks = [c for c in all_chunks if c["source"] == source][:chunks_per_source]
    sample_chunks.extend(source_chunks)
print(f"Sampling from sources: {target_sources}")
print(f"Extracting triples from {len(sample_chunks)} sample chunks...")

for i, chunk in enumerate(sample_chunks):
    triples = extract_triples(chunk["text"])
    print(f"Chunk {i+1}: found {len(triples)} triples")
    for triple in triples:
        if len(triple) == 3:
            subj, rel, obj = triple
            subj = subj.strip().lower()
            obj = obj.strip().lower()
            graph.add_edge(subj, obj, relationship=rel)

print(f"\nGraph built: {graph.number_of_nodes()} entities, {graph.number_of_edges()} relationships")

wolf_chunks_used = [c for c in sample_chunks if c["source"] == "wiki_wolf.txt"]
print(f"\nActual wiki_wolf.txt chunks in sample: {len(wolf_chunks_used)}")

wolf_related_edges = [
    (u, v, d) for u, v, d in graph.edges(data=True) if "wolf" in u.lower() or "wolf" in v.lower()
]
print(f"\nAll edges mentioning 'wolf' ({len(wolf_related_edges)} found):")
for u, v, d in wolf_related_edges[:20]:
    print(f"  {u} --[{d['relationship']}]--> {v}")
print("\nSample edges:")
for u, v, data in list(graph.edges(data=True))[:10]:
    print(f"  {u} --[{data['relationship']}]--> {v}")


def find_related(entity, max_hops=2):
    if entity not in graph:
        matches = [n for n in graph.nodes if entity.lower() in n.lower()]
        if not matches:
            return f"No entity found matching '{entity}'"
        entity = matches[0]

    print(f"\nExploring from: {entity}")
    for target in graph.nodes:
        if target == entity:
            continue
        try:
            path = nx.shortest_path(graph, entity, target)
            if 1 < len(path) <= max_hops + 1:
                relationships = []
                for i in range(len(path) - 1):
                    edge_data = graph.get_edge_data(path[i], path[i + 1])
                    relationships.append(
                        f"{path[i]} --[{edge_data['relationship']}]--> {path[i+1]}"
                    )
                print("  " + " | ".join(relationships))
        except nx.NetworkXNoPath:
            continue


find_related("dogs", max_hops=3)
print("\nOutgoing edges from 'wolves' (rechecking with expanded sample):")
for u, v, data in graph.out_edges("wolves", data=True):
    print(f"  {u} --[{data['relationship']}]--> {v}")
print(f"(wolves now has {graph.out_degree('wolves')} outgoing edges)")

print("\nOutgoing edges from 'wolves':")
for u, v, data in graph.out_edges("wolves", data=True):
    print(f"  {u} --[{data['relationship']}]--> {v}")
print(f"(wolves has {graph.out_degree('wolves')} outgoing edges)")

print("\nOutgoing edges from 'domesticated mammals':")
for u, v, data in graph.out_edges("domesticated mammals", data=True):
    print(f"  {u} --[{data['relationship']}]--> {v}")
print(f"(domesticated mammals has {graph.out_degree('domesticated mammals')} outgoing edges)")
