import networkx as nx

def main():
    file_name = 'data/wikidata/train.txt'
    KG = nx.Graph()

    with open(file_name, 'r') as f:
        for line in f.readlines():
            ws = line.strip().split("\t")
            if len(ws) < 3:
                continue
        
            h = ws[0]
            p = ws[1]
            t = ws[2]

            p_reverse = p+"_reverse"
            KG.add_edges_from([(h, t, {'rel': p})])
    
    ppr1 = nx.pagerank(KG)

    out_name = 'data/wikidata/page_rank_score.txt'
    with open(out_name, 'w') as f:
        for k, v in ppr1.items():
            f.write(k.strip() + "\t" + str(v) + "\n")
    a = 0


main()