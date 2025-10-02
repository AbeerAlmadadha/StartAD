# Data augmentation script for Arabic teacher notes
import pandas as pd
import numpy as np
import random
import re

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Arabic synonyms dictionary for data augmentation
ARABIC_SYNONYMS = {
    # Student words
    "الطالب": ["التلميذ", "الدارس", "الطفل"],
    "الطالبة": ["التلميذة", "الدارسة", "الطفلة"],

    # Positive performance words
    "متفوقة": ["ممتازة", "متميزة", "رائعة", "مبدعة"],
    "متفوق": ["ممتاز", "متميز", "رائع", "مبدع"],
    "مشاركة": ["متفاعلة", "نشطة", "فعالة"],
    "يشارك": ["يتفاعل", "ينشط", "يساهم"],
    "جيدة": ["ممتازة", "رائعة", "مميزة"],
    "جيد": ["ممتاز", "رائع", "مميز"],
    "دافعية": ["حماس", "رغبة", "شغف"],
    "حماس": ["دافعية", "شغف", "رغبة"],

    # Moderate performance words
    "يحتاج": ["يتطلب", "يستدعي", "يفتقر إلى"],
    "تحتاج": ["تتطلب", "تستدعي", "تفتقر إلى"],
    "مساعدة": ["دعم", "عون", "مساندة"],
    "إضافية": ["أكثر", "زائدة", "أخرى"],
    "أحيانًا": ["بعض الأوقات", "أحياناً", "في بعض الأحيان"],
    "يتشتت": ["يفقد التركيز", "يتشتت انتباهه", "ينشغل"],
    "تتردد": ["تتأخر", "تترفع", "تخاف"],

    # Negative performance words
    "صعوبة": ["مشكلة", "تحدي", "عقبة"],
    "صعوبات": ["مشاكل", "تحديات", "عقبات"],
    "يعاني": ["يواجه", "يتصارع مع", "يجد صعوبة في"],
    "تعاني": ["تواجه", "تتصارع مع", "تجد صعوبة في"],
    "قلق": ["توتر", "خوف", "انزعاج"],
    "إحباط": ["يأس", "ضيق", "انزعاج"],
    "ضعف": ["نقص", "قصور", "عجز"],
    "فقدان": ["نقص", "غياب", "انعدام"],

    # Academic subjects
    "الرياضيات": ["الحساب", "الأرقام"],
    "العلوم": ["المواد العلمية"],
    "القراءة": ["المطالعة"],
    "الكتابة": ["التأليف", "الإنشاء"],

    # General words
    "كبيرة": ["شديدة", "عظيمة", "واضحة"],
    "كبير": ["شديد", "عظيم", "واضح"],
    "متابعة": ["مراقبة", "ملاحظة", "رعاية"],
    "تقدم": ["تحسن", "تطور", "نمو"],
    "ينجز": ["يكمل", "ينهي", "يحقق"],
    "تنجز": ["تكمل", "تنهي", "تحقق"],
    "واجباته": ["مهامه", "تكليفاته", "فروضه"],
    "واجباتها": ["مهامها", "تكليفاتها", "فروضها"]
}

def replace_synonyms(text, probability=0.3):
    """Replace words with their Arabic synonyms"""
    words = text.split()
    new_words = []

    for word in words:
        if random.random() < probability and word in ARABIC_SYNONYMS:
            # Choose a random synonym
            synonym = random.choice(ARABIC_SYNONYMS[word])
            new_words.append(synonym)
        else:
            new_words.append(word)

    return " ".join(new_words)

def add_intensity_modifiers(text, label):
    """Add intensity modifiers based on the label"""
    intensity_words = {
        'high': ['جداً', 'بشدة', 'كثيراً', 'للغاية'],
        'moderate': ['نوعاً ما', 'إلى حد ما', 'قليلاً'],
        'low': ['تماماً', 'بوضوح', 'جداً']
    }

    if random.random() < 0.4:  # 40% chance to add modifier
        modifier = random.choice(intensity_words[label])
        # Add modifier at the end or before adjectives
        if any(word in text for word in ['صعوبة', 'متفوقة', 'جيدة', 'ممتازة']):
            text = text.replace('صعوبة', f'صعوبة {modifier}')
            text = text.replace('متفوقة', f'متفوقة {modifier}')
            text = text.replace('جيدة', f'جيدة {modifier}')
            text = text.replace('ممتازة', f'ممتازة {modifier}')
        else:
            text = f"{text} {modifier}"

    return text

def rephrase_sentence(text, label):
    """Create variations by rephrasing sentences"""
    variations = []

    # Original text
    variations.append(text)

    # Add context variations
    if label == 'high':
        prefixes = ['للأسف', 'مع الأسف', 'يبدو أن']
        if random.random() < 0.3:
            variations.append(f"{random.choice(prefixes)} {text}")
    elif label == 'low':
        prefixes = ['لحسن الحظ', 'من الجيد أن', 'يسعدني أن']
        if random.random() < 0.3:
            variations.append(f"{random.choice(prefixes)} {text}")

    # Add time context
    time_contexts = ['هذا الأسبوع', 'مؤخراً', 'في الآونة الأخيرة', 'خلال الفترة الماضية']
    if random.random() < 0.2:
        variations.append(f"{text} {random.choice(time_contexts)}")

    return random.choice(variations)

def augment_data(df, target_samples_per_class=100):
    """Augment the dataset to reach target samples per class"""
    augmented_data = []

    for label in df['label'].unique():
        label_data = df[df['label'] == label]
        current_count = len(label_data)
        target_count = target_samples_per_class

        print(f"Processing {label} class: {current_count} → {target_count} samples")

        # Add original data
        for _, row in label_data.iterrows():
            augmented_data.append({
                'teacher_note': row['teacher_note'],
                'label': row['label'],
                'augmented': False
            })

        # Generate augmented samples
        samples_needed = target_count - current_count
        generated = 0

        while generated < samples_needed:
            # Pick a random original sample
            original_row = label_data.sample(1).iloc[0]
            original_text = original_row['teacher_note']

            # Apply augmentation techniques
            augmented_text = original_text

            # Apply synonym replacement
            if random.random() < 0.7:
                augmented_text = replace_synonyms(augmented_text)

            # Add intensity modifiers
            if random.random() < 0.5:
                augmented_text = add_intensity_modifiers(augmented_text, label)

            # Rephrase sentence
            if random.random() < 0.4:
                augmented_text = rephrase_sentence(augmented_text, label)

            # Only add if it's different from original
            if augmented_text != original_text:
                augmented_data.append({
                    'teacher_note': augmented_text,
                    'label': label,
                    'augmented': True
                })
                generated += 1

    return pd.DataFrame(augmented_data)

def main():
    print("Starting data augmentation for Arabic teacher notes...")

    # Load original data
    df = pd.read_csv("data/teacher_notes_labeled.csv")
    print(f"Original dataset: {len(df)} samples")
    print("Original distribution:")
    print(df['label'].value_counts())

    # Augment data
    augmented_df = augment_data(df, target_samples_per_class=100)

    print(f"\nAugmented dataset: {len(augmented_df)} samples")
    print("New distribution:")
    print(augmented_df['label'].value_counts())

    print(f"Augmented samples: {len(augmented_df[augmented_df['augmented']==True])}")
    print(f"Original samples: {len(augmented_df[augmented_df['augmented']==False])}")

    # Save augmented dataset
    output_file = "data/teacher_notes_augmented.csv"
    augmented_df[['teacher_note', 'label']].to_csv(output_file, index=False)
    print(f"\nAugmented dataset saved to: {output_file}")

    # Show some examples
    print("\nSample augmented examples:")
    for label in ['low', 'moderate', 'high']:
        print(f"\n{label.upper()} class examples:")
        label_samples = augmented_df[(augmented_df['label']==label) & (augmented_df['augmented']==True)]
        for i, (_, row) in enumerate(label_samples.head(3).iterrows()):
            print(f"  {i+1}. {row['teacher_note']}")

if __name__ == "__main__":
    main()
