import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union

from .prompt import extract_answer_dk, answer_question_withoutcot
from .tools import get_response, clear_json

EvaluationResult = Dict[str, Any]


def simulate(
    language: str = 'English',
    question: Optional[str] = None,
    model_list: Optional[List[str]] = None,
    choices: Optional[List[str]] = None,
    ground_truth: Optional[str] = None,
    max_tries: int = 5,
    return_details: bool = False,
) -> Union[float, EvaluationResult]:
    model_list = model_list or []
    choices = choices or []
    question_prompt = {
        'Chinese': '请完成以下单选题，并严格从给定的选项中选择一个最符合题意的答案：',
        'English': 'Please complete the following multiple-choice question and select the one option that best fits the given context. Strictly choose from the provided options without adding explanations or modifying the choices: ',
        'French': 'Veuillez répondre à la question à choix unique suivante et sélectionner l’option qui correspond le mieux au contexte donné. Veuillez strictement choisir parmi les options fournies, sans ajouter d’explications ni modifier les choix :',
        'Spanish': 'Por favor, completa la siguiente pregunta de opción múltiple y selecciona la opción que mejor se ajuste al contexto dado. Elige estrictamente entre las opciones proporcionadas sin añadir explicaciones ni modificar las opciones:',
        'Ukrainian':'Будь ласка, завершіть наступне питання з вибором відповіді та виберіть один варіант, який найкраще відповідає даному контексту. Строго обирайте з наданих варіантів без додавання пояснень чи зміни варіантів:',
        'Arabic':'يرجى إكمال سؤال الاختيار من متعدد التالي واختيار الخيار الوحيد الذي يناسب السياق المعطى بشكل أفضل. اختر بدقة من الخيارات المقدمة دون إضافة تفسيرات أو تعديل الخيارات:',
        'Japanese':'次の選択式質問を完成させ、与えられた文脈に最も適した選択肢を1つ選んでください。説明を追加したり、選択肢を変更したりせずに、提供された選択肢から厳密に選んでください:',
        'Korean':'다음 객관식 질문을 완성하고 주어진 상황에 가장 적합한 옵션 하나를 선택하세요. 설명을 추가하거나 선택지를 수정하지 말고 제공된 옵션에서만 엄격히 선택하세요:',
        'German':'Bitte vervollständigen Sie die folgende Multiple-Choice-Frage und wählen Sie die Option aus, die am besten zum gegebenen Kontext passt. Wählen Sie strikt aus den vorgegebenen Optionen, ohne Erklärungen hinzuzufügen oder die Auswahlmöglichkeiten zu ändern:',
        'Italian':f"""Si prega di completare la seguente domanda a scelta multipla e selezionare l'opzione che meglio si adatta al contesto fornito. Scegli esclusivamente tra le opzioni fornite, senza aggiungere spiegazioni o modificare le scelte:""",
        'Portuguese':'Por favor, complete a seguinte pergunta de múltipla escolha e selecione a opção que melhor se adapta ao contexto fornecido. Escolha estritamente entre as opções fornecidas, sem adicionar explicações ou modificar as alternativas:',
        'Bengali':'দয়া করে নিম্নলিখিত বহু নির্বাচনী প্রশ্নটি সম্পূর্ণ করুন এবং প্রদত্ত প্রসঙ্গের সাথে সবচেয়ে ভাল মানানসই একটি বিকল্প নির্বাচন করুন। ব্যাখ্যা যোগ করা বা বিকল্পগুলি পরিবর্তন করা ছাড়াই কেবল সরবরাহকৃত বিকল্পগুলির মধ্য থেকে কঠোরভাবে নির্বাচন করুন।:',
        'Hindi':'कृपया निम्नलिखित बहुविकल्पीय प्रश्न को पूरा करें और दिए गए संदर्भ के अनुसार सबसे उपयुक्त विकल्प का चयन करें। केवल प्रदान किए गए विकल्पों में से ही चयन करें, बिना किसी व्याख्या जोड़े या विकल्पों में बदलाव किए।:',
        'Hebrew':'אנא השלם את שאלת הבחירה המרובה הבאה ובחר את האפשרות שהכי מתאימה להקשר הנתון. בחר אך ורק מתוך האפשרויות שסופקו, מבלי להוסיף הסברים או לשנות את האפשרויות:',
        'Amharic':'እባኮትን የተከታታይውን በርካታ ምርጫ ጥያቄ ሙሉ አድርጉ እና ለተሰጠው ሁኔታ በጣም የሚስማማውን አንድ አማራጭ ይምረጡ። ማብራሪያ ሳታከሉ ወይም አማራጮቹን ሳትለዋወጡ በተሰጡት አማራጮች ብቻ ይምረጡ።:',
        'Yoruba':'Jọwọ parí ìbéèrè ìyanjú-yàn lábẹ́ yìí kí o sì yan aṣayan kan tó bá àkójọ yìí mu jù. Yàn láti inú àwọn àṣàyàn tó wà nìkan, láìfikún àlàyé tàbí yí àwọn àṣàyàn padà:',
        'Swahili':'Tafadhali kamilisha swali lifuatalo la chaguo nyingi na uchague jibu moja linalofaa zaidi kwa muktadha uliotolewa. Chagua tu kutoka kwa chaguo zilizotolewa bila kuongeza maelezo au kubadilisha chaguo hizo:',
        'Zulu':'Sicela uqede umbuzo olandelayo wokukhetha okuningi bese ukhetha inketho eyodwa ehambisana kakhulu nomongo onikeziwe. Khetha ngokuqinile ezinkethweni ezinikeziwe ngaphandle kokungeza izincazelo noma ukuguqula izinketho:'
    }
    prompt = question_prompt[language]+answer_question_withoutcot(question=question, choices=choices)
    model_results: Dict[str, EvaluationResult] = {}

    def get_answer_dk(model: str = 'gpt-4o') -> EvaluationResult:
        result: EvaluationResult = {
            'status': 'request_failed',
            'raw_response': None,
            'extracted_answer': None,
            'correct': False,
            'score': 0.0,
            'extractor_model': os.getenv('ANSWER_EXTRACT_MODEL', 'gpt-4o-mini'),
        }
        response = get_response(model=model, prompt=prompt, temperature=0.0001)
        if response is None:
            print(f"{model}: request failed after retries")
            return result

        result['raw_response'] = response
        extract_prompt = extract_answer_dk(question=question, answer=response, choices=choices)
        extractor_model = result['extractor_model']
        extraction_response = get_response(
            model=extractor_model,
            prompt=extract_prompt,
            temperature=0.0001,
        )
        if extraction_response is None:
            result['status'] = 'extraction_request_failed'
            print(f"{model}: answer extraction failed after retries with {extractor_model}")
            return result

        cleaned_answer = clear_json(extraction_response)
        try:
            parsed_answer = json.loads(cleaned_answer)
            answer = parsed_answer['final_answer']
        except (json.JSONDecodeError, KeyError, TypeError):
            result['status'] = 'invalid_extraction_json'
            result['extraction_response'] = extraction_response
            print(f"{model}: failed to extract answer")
            return result

        result['extracted_answer'] = answer
        if answer not in choices:
            result['status'] = 'out_of_choices'
            print(f"{model}: extracted answer is out of choices")
            return result

        result['status'] = 'success'
        result['correct'] = answer == ground_truth
        result['score'] = 1.0 if result['correct'] else 0.0
        print(f"{model}: {answer}")
        return result

    with ThreadPoolExecutor() as executor:
        future_answers = {
            executor.submit(get_answer_dk, model): model
            for model in model_list
        }
        for future in as_completed(future_answers):
            model = future_answers[future]
            model_results[model] = future.result()

    successful_results = [
        result for result in model_results.values()
        if result['status'] == 'success'
    ]
    if successful_results:
        correct_rate = sum(result['score'] for result in successful_results) / len(successful_results)
    else:
        correct_rate = 0.0

    if return_details:
        return {
            'score': correct_rate,
            'models': model_results,
        }
    return correct_rate
