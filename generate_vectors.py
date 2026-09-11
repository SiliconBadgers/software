import argparse
import json
from pathlib import Path
import random
from reference import step

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    contract=json.loads(a.contract.read_text())
    expected={'id':'siliconbadgers.mac.v0','operand_bits':8,'accumulator_bits':32,'signed':True,'overflow':'wrap','clear_priority':'before_enable','reset':'asynchronous_active_low','vector_columns':['a','b','enable','clear','expected_acc']}
    if contract != expected:
        p.error('Unsupported contract; update model and consumers explicitly.')
    rng=random.Random(42)
    cases=[(0,0,0,1),(-128,-128,1,0),(127,-128,1,0),(127,127,0,0),(127,127,1,1)]
    cases += [(rng.randrange(-128,128),rng.randrange(-128,128),int(i%7!=0),int(i%31==0)) for i in range(256)]
    acc=0
    rows=[]
    for av,bv,en,clr in cases:
        acc=step(acc,av,bv,en,clr)
        rows.append(f'{av} {bv} {en} {clr} {acc}')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text('\n'.join(rows)+'\n')
    print(f'PASS generated {len(rows)} golden vectors for {contract["id"]}')

if __name__=='__main__': main()
