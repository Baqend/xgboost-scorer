interface BoosterNode {
  nodeid: number,
  depth: number,
  split: string,
  split_condition: number,
  yes: number,
  no: number,
  missing: number,
  children: Array<BoosterNode | BoosterLeaf>
}

interface BoosterLeaf {
  nodeid: number,
  leaf: number
}

type Booster = BoosterNode; // The root of the tree
type XGBoostModel = Array<Booster>;

function sigmoid(x: number) {
  return 1 / (1 + Math.pow(Math.E, -x));
}

function isLeaf(node: BoosterNode | BoosterLeaf): node is BoosterLeaf {
  return (node as BoosterLeaf).leaf !== undefined;
}

export default class Scorer {
  model?: XGBoostModel;

  static async create(model: XGBoostModel) {
    const scorer = new Scorer
    scorer.model = model;
    return scorer;
  }

  scoreSingleInstance(features: Record<string, number>) {
    if (!this.model) {
      throw new Error(`Scorer not initialized, create a scorer using Scorer.create() only`)
    }
    const totalScore: number =
      this.model
        .map((booster: Booster) => {
          let currNode: BoosterNode | BoosterLeaf = booster;
          while (!isLeaf(currNode)) {
            const splitFeature = currNode.split;
            let nextNodeId: number;
            if (features[splitFeature] !== undefined) {
              const conditionResult = features[splitFeature] < currNode.split_condition;
              nextNodeId = conditionResult ? currNode.yes : currNode.no;
            } else {
              // Missing feature
              nextNodeId = currNode.missing;
            }
            const nextNode: BoosterNode | BoosterLeaf | undefined =
              currNode.children.find(child => child.nodeid === nextNodeId);
            if (nextNode === undefined) {
              throw new Error(`Invalid model JSON, missing node ID: ${nextNodeId}`)
            }
            currNode = nextNode;
          }
          return currNode.leaf;
        })
        .reduce((score, boosterScore) => score + boosterScore, 0.0)
    return sigmoid(totalScore);
  }

  async score(input: object | Array<object>): Promise<Array<number> | number> {
    if (typeof input !== "object") {
      throw new Error(`Invalid input to score method: ${input}, expected string or object, was ${typeof input}`)
    }

    // Scoring a single instance or array of instances
    if (Array.isArray(input)) {
      return (input as Array<object>).map(en => this.scoreSingleInstance(en as Record<string, number>));
    } else {
      return this.scoreSingleInstance(input as Record<string, number>);
    }
  }
}
