

def main(filename):
    dataset = filename.split('.')[0]
    problems_plain_text = "{}.plain.txt".format(dataset)
    mapping = open("{}.mapping".format(dataset))
    mapping = [item.strip().split("|") for item in mapping]

    with open(problems_plain_text, "w") as f:
        with open(filename,'r') as inputfile:
            for line in inputfile:
                s=line
                for item in mapping:
                    s = s.replace('<%s>' % item[0], '<%s>' % item[1])
                print(s.strip(), file=f)


if __name__ == "__main__":
    main('ChicagoBike.txt')
