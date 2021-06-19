import getopt, sys

def get_args(argv=sys.argv[1:]):
    language = ''
    representation = ''
    model=''
    people = None
    segments = None
    try:
        opts, args = getopt.getopt(argv,"h:l:r:p:s:m:",["language=","representation=","people=","segments=","model="])
    except getopt.GetoptError:
        print('test.py -l <language> -r <representation> -p <people> -s <segments>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py -l <language> -r <representation> -p <people> -s <segments>')
            sys.exit()
        elif opt in ("-l", "--language"):
            language = arg
        elif opt in ("-r", "--representation"):
            representation = arg
        elif opt in ("-s", "--segments"):
            segments = int(arg)
        elif opt in ("-p", "--people"):
            people = int(arg)
        elif opt in ("-m", "--model"):
            model = arg

    args = {
        'language': language,
        'representation': representation,
        'people': people,
        'segments': segments,
        'model': model
    }

    print('ARGS:', args)

    return args