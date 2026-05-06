import sys
from phase1_ludii_rag.ludii_nlp_pipeline import LudiiNLPPipeline
from phase1_ludii_rag.damaged_rules_generator import DamagedRulesGenerator

def main():
    if len(sys.argv) < 2:
        print('Usage: setup / generate-dataset / search / restore / evaluate / stats')
        return
    cmd = sys.argv[1]
    args = sys.argv[2:]
    if cmd == 'setup':
        LudiiNLPPipeline().setup(reset_db=True)
    elif cmd == 'generate-dataset':
        DamagedRulesGenerator().generate_dataset()
    elif cmd == 'search':
        pipeline = LudiiNLPPipeline()
        results = pipeline.rag.retrieve_similar_rules(args[0], top_k=3)
        for i, r in enumerate(results, 1):
            print(str(i) + ' ' + r['game'] + ' sim=' + str(r['similarity']))
            print(r['text'][:300])
    elif cmd == 'restore':
        pipeline = LudiiNLPPipeline()
        game = args[1] if len(args) > 1 else None
        print(pipeline.restore_rule(args[0], game=game))
    elif cmd == 'evaluate':
        print(LudiiNLPPipeline().evaluate_on_damaged_dataset())
    elif cmd == 'stats':
        print(LudiiNLPPipeline().rag.stats())

if __name__ == '__main__':
    main()
